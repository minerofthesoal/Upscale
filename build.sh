#!/bin/bash
# Builds AIUpscaleTweak.dylib for iOS (arm64).
# Must run on macOS with Xcode + iOS SDK installed. This won't run in a
# plain Linux/CI shell — there's no iOS SDK to link against off-device.
set -euo pipefail

SDK=$(xcrun --sdk iphoneos --show-sdk-path)
OUT="AIUpscaleTweak.dylib"

xcrun -sdk iphoneos clang \
  -arch arm64 \
  -isysroot "$SDK" \
  -miphoneos-version-min=17.0 \
  -dynamiclib \
  -fobjc-arc \
  -framework Foundation \
  -framework UIKit \
  -framework QuartzCore \
  -framework Metal \
  -framework MetalFX \
  -o "$OUT" \
  AIUpscaleTweak.m

# Ad-hoc sign so it loads under LiveContainer's re-signing pass.
codesign -s - "$OUT"

echo "Built $OUT"
echo "Copy it into LiveContainer's Tweaks folder (global, or scoped to one app)."
