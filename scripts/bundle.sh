#!/usr/bin/env bash
# Builds GazeFuse.app (release) into ./dist and ad-hoc signs it.
# Usage: scripts/bundle.sh [--install]   (--install copies to ~/Applications)
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release --product GazeFuse --disable-sandbox
BIN="$(swift build -c release --show-bin-path --disable-sandbox)/GazeFuse"
APP=dist/GazeFuse.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/GazeFuse"
cp Resources/Info.plist "$APP/Contents/Info.plist"
# Ad-hoc signature with a stable identifier. NOTE: macOS ties the Accessibility grant to the
# code signature; after a rebuild you may need to toggle GazeFuse off/on in Accessibility settings.
codesign --force --sign - --identifier io.github.jazelly.gazefuse \
  --entitlements Resources/GazeFuse.entitlements "$APP"
codesign --verify --verbose "$APP"
echo "Built $APP"
if [[ "${1:-}" == "--install" ]]; then
  mkdir -p ~/Applications && rm -rf ~/Applications/GazeFuse.app && cp -R "$APP" ~/Applications/
  echo "Installed to ~/Applications/GazeFuse.app"
fi
