# Releasing Mishkefet (personal team)

Everything here signs and uploads as **Itay Inbar, team `65MGR94YVU`**, using the API key in
`~/.appstoreconnect/itay-personal-team/`. Nothing here uses the Henia team (`87SV5ZQ3H8`).

## One-time (done)
- Bundle ID `com.itayinbar.mishkefet` registered through the API.
- App record created in App Store Connect: id `6823990087`, "משקפת - חילוץ כתב יד".
- Xcode signed in (Settings, Accounts) with the Apple ID on team 65MGR94YVU; signing uses it.
- After the first upload: **App Privacy**, **Get Started**, "No, we do not collect data", Publish.

## Every release
```sh
ios/release/release-testflight.sh        # test, archive, sign, validate, upload, TestFlight (Itay only)
```
Submitting for review is a deliberate manual step in App Store Connect, never scripted.
