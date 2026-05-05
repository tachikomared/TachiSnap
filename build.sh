#!/usr/bin/env bash
set -e

echo "TachiSnap - Pixel Snapper for animation pixel artists"
echo "Made by TachikomaRed and smolemaru"
echo "====================================================="

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

echo "▶  Preparing static dist..."
mkdir -p dist
cp -r web/* dist/
cp -r pkg dist/pkg
rm -f dist/pkg/.gitignore dist/pkg/package.json

echo ""
echo "✅ Build complete!"
echo ""
echo "   Serve the project root, e.g.:"
echo "   python3 -m http.server 8080"
echo "   then open http://localhost:8080"
echo ""
echo "   Static deploy output:"
echo "   dist/"
