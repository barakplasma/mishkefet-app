#!/bin/sh
# Fetch the pinned Mishkefet-v1 app artifacts (HF itayinbar/Mishkefet-v1, onnx/),
# verify SHA-256, cache them, and optionally copy them into a target directory.
#   tools/model/fetch-model.sh [target-dir]
# Used by the Xcode pre-build script, the release script and the tests.
set -eu
REPO="itayinbar/Mishkefet-v1"
REV="16380452d62839f369c36f17312c94d5d1af1768"
CACHE="${MISHKEFET_MODEL_CACHE:-$HOME/.cache/mishkefet/models/$(echo $REV | cut -c1-8)}"
mkdir -p "$CACHE"
while read -r sha name; do
  f="$CACHE/$name"
  if [ ! -f "$f" ] || [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" != "$sha" ]; then
    echo "fetching $name"
    curl -fL --retry 4 -o "$f.part" "https://huggingface.co/$REPO/resolve/$REV/onnx/$name"
    got="$(shasum -a 256 "$f.part" | cut -d' ' -f1)"
    [ "$got" = "$sha" ] || { echo "sha256 mismatch for $name: $got" >&2; rm -f "$f.part"; exit 1; }
    mv "$f.part" "$f"
  fi
done <<LIST
d7cf4fb3c89016c52038ea6405e6080da62742eb75a589b6ad6d5388bdb48630 backbone.fp16.onnx
0460fa2c2abf6033efb192dce3da639513878f191d3da302db6b665a493dc687 head.bin
085190280be0bd967cb0013ef29a6e2d3451d425a8a0f3d30344d8092e0763f4 charlm.bin
25de5455bc73a79d7cc63c205e0bc137ceab0f9c2dfe1496e93aef8abc7ebe61 words.bin
e366e47b758c0174cf281a502d4e7e8ec2953795e7b29506bff0292a44bfa9a1 charset.json
LIST
if [ $# -ge 1 ]; then
  mkdir -p "$1"
  for n in backbone.fp16.onnx head.bin charlm.bin words.bin charset.json; do
    cmp -s "$CACHE/$n" "$1/$n" 2>/dev/null || cp "$CACHE/$n" "$1/$n"
  done
fi
echo "$CACHE"
