#!/usr/bin/env bash
# Archives Mishkefet, signs it for the App Store with the PERSONAL team (Itay Inbar, 65MGR94YVU),
# uploads it to App Store Connect and makes it available in TestFlight to Itay only.
#
#   ios/release/release-testflight.sh
#
# Needs: ~/.appstoreconnect/itay-personal-team/{AuthKey_<id>.p8,key_id,issuer} (upload + TestFlight),
# Xcode signed in to the Apple ID on team 65MGR94YVU (signing), and the
# "Apple Distribution: Itay Inbar (65MGR94YVU)" identity in the login keychain. The app record
# (bundle id com.itayinbar.mishkefet) must already exist in App Store Connect.
set -euo pipefail
cd "$(dirname "$0")/.."

TEAM_ID="65MGR94YVU"          # personal team, never 87SV5ZQ3H8 (Henia/Amir)
BUNDLE_ID="com.itayinbar.mishkefet"
DIR="${HOME}/.appstoreconnect/itay-personal-team"
KEY_ID="$(cat "${DIR}/key_id")"
ISSUER="$(cat "${DIR}/issuer")"
KEY_PATH="${DIR}/AuthKey_${KEY_ID}.p8"
[ -f "$KEY_PATH" ] || { echo "Missing $KEY_PATH"; exit 1; }
security find-identity -v -p codesigning | grep -q "Apple Distribution: .*(${TEAM_ID})" \
  || { echo "No Apple Distribution identity for ${TEAM_ID} in the keychain"; exit 1; }
# Signing (archive/export) goes through the Apple ID signed in to Xcode (Settings, Accounts):
# the personal API key can call App Store Connect but Xcode rejects it for provisioning.
# MK_XCODE_KEY=1 switches to key-based signing (needs an Admin-role key).
if [ "${MK_XCODE_KEY:-0}" = "1" ]; then
  AUTH=(-allowProvisioningUpdates -authenticationKeyPath "$KEY_PATH" -authenticationKeyID "$KEY_ID" -authenticationKeyIssuerID "$ISSUER")
else
  AUTH=(-allowProvisioningUpdates)
fi

BUILD="$(date -u +%Y%m%d%H%M)"   # build numbers only ever increase
OUT="build/release"
ARCHIVE="${OUT}/Mishkefet-${BUILD}.xcarchive"
EXPORT="${OUT}/export-${BUILD}"
mkdir -p "$OUT"

echo "== Model files"
../tools/model/fetch-model.sh Mishkefet/Model >/dev/null

echo "== Project"
xcodegen generate >/dev/null

echo "== Tests"
(cd MishkefetCore && swift test -q)
xcodebuild -project Mishkefet.xcodeproj -scheme Mishkefet -destination "platform=iOS Simulator,name=${TEST_SIMULATOR:-iPhone 17 Pro}" \
  CODE_SIGNING_ALLOWED=NO test -quiet

echo "== Archive build ${BUILD}"
xcodebuild -project Mishkefet.xcodeproj -scheme Mishkefet -configuration Release -destination 'generic/platform=iOS' \
  -archivePath "$ARCHIVE" "${AUTH[@]}" DEVELOPMENT_TEAM="$TEAM_ID" CURRENT_PROJECT_VERSION="$BUILD" archive -quiet

cat > "${OUT}/ExportOptions-${BUILD}.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>method</key><string>app-store-connect</string>
  <key>destination</key><string>export</string>
  <key>teamID</key><string>${TEAM_ID}</string>
  <key>signingStyle</key><string>automatic</string>
  <key>uploadSymbols</key><true/>
  <key>manageAppVersionAndBuildNumber</key><false/>
</dict></plist>
PLIST

echo "== Export (App Store signing)"
xcodebuild -exportArchive -archivePath "$ARCHIVE" -exportOptionsPlist "${OUT}/ExportOptions-${BUILD}.plist" \
  -exportPath "$EXPORT" "${AUTH[@]}" -quiet
IPA="$(ls "${EXPORT}"/*.ipa | head -1)"
# Every bundle in the IPA must carry the personal team's distribution signature.
CHECK="$(mktemp -d)"; unzip -q "$IPA" -d "$CHECK"
for bundle in "$CHECK"/Payload/*.app "$CHECK"/Payload/*.app/PlugIns/*.appex; do
  sig="$(codesign -dvv "$bundle" 2>&1 || true)"
  case "$sig" in
    *"TeamIdentifier=${TEAM_ID}"*"Authority=Apple Distribution"*|*"Authority=Apple Distribution"*"TeamIdentifier=${TEAM_ID}"*) ;;
    *) echo "$(basename "$bundle") is not distribution-signed by ${TEAM_ID}; refusing to upload"; exit 1 ;;
  esac
done
# iPhone only: an iPad-capable build makes App Store Connect demand 13-inch iPad screenshots.
family="$(plutil -extract UIDeviceFamily json -o - "$CHECK"/Payload/*.app/Info.plist)"
[ "$family" = "[1]" ] || { echo "UIDeviceFamily is $family, expected [1] (iPhone only); refusing to upload"; exit 1; }
rm -rf "$CHECK"
du -h "$IPA"

echo "== Validate + upload"
xcrun altool --validate-app -f "$IPA" -t ios --apiKey "$KEY_ID" --apiIssuer "$ISSUER"
xcrun altool --upload-app -f "$IPA" -t ios --apiKey "$KEY_ID" --apiIssuer "$ISSUER"

echo "== Waiting for processing, then TestFlight"
node --no-warnings release/asc-testflight.mjs "$BUILD" "$BUNDLE_ID"
echo "Mishkefet build ${BUILD} uploaded."
