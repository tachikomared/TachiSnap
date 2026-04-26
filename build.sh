#!/usr/bin/env bash
set -e

echo "🎮 Pixel Snapper — Build Script"
echo "================================"

# Check wasm-pack
if ! command -v wasm-pack &>/dev/null; then
  echo "⚠  wasm-pack not found. Install it:"
  echo "   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
  exit 1
fi

echo "▶  Building WASM module (release)..."
wasm-pack build --target web --out-dir pkg --release

echo "▶  Copying web files..."
cp -r web/* .

echo ""
echo "✅ Build complete!"
echo ""
echo "   Serve the project root, e.g.:"
echo "   python3 -m http.server 8080"
echo "   then open http://localhost:8080"
