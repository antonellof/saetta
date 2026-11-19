#!/bin/sh
# saetta installer. Downloads the release binary for Apple Silicon, checks its SHA-256
# and installs it into $SAETTA_PREFIX/bin (default ~/.local).
#   curl -fsSL https://github.com/antonellof/saetta/releases/latest/download/install.sh | sh
# SAETTA_VERSION=0.7.0 pins a release (default: the latest).
set -eu
REPO=antonellof/saetta
ASSET=saetta-macos-arm64.tar.gz
PREFIX=${SAETTA_PREFIX:-$HOME/.local}
VERSION=${SAETTA_VERSION:-latest}

say() { printf 'saetta: %s\n' "$*"; }
die() { printf 'saetta: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] || die "macOS only (Apple Silicon)."
[ "$(uname -m)" = arm64 ] || die "Apple Silicon (arm64) only."
os=$(sw_vers -productVersion)
major=${os%%.*}; rest=${os#*.}; minor=${rest%%.*}
[ "$rest" = "$os" ] && minor=0
if [ "$major" -lt 27 ] || { [ "$major" -eq 27 ] && [ "$minor" -lt 2 ]; }; then
  die "needs macOS 27.2 or later (this is $os)."
fi
mem=$(( $(sysctl -n hw.memsize) / 1073741824 ))
[ "$mem" -ge 128 ] || say "warning: built for 128 GB of memory, this Mac has $mem GB; the model may not fit."

if [ "$VERSION" = latest ]; then base="https://github.com/$REPO/releases/latest/download"
else base="https://github.com/$REPO/releases/download/v${VERSION#v}"; fi

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
say "downloading $base/$ASSET"
curl -fsSL --connect-timeout 15 -o "$tmp/$ASSET" "$base/$ASSET" || die "download failed."
curl -fsSL --connect-timeout 15 -o "$tmp/$ASSET.sha256" "$base/$ASSET.sha256" || die "checksum download failed."
(cd "$tmp" && shasum -a 256 -c "$ASSET.sha256" >/dev/null) || die "checksum mismatch, not installed."
tar -xzf "$tmp/$ASSET" -C "$tmp"
src=$(find "$tmp" -maxdepth 1 -type d -name 'saetta-*' | head -1)
[ -x "$src/bin/saetta" ] || die "unexpected archive layout."

mkdir -p "$PREFIX/bin"
install -m 755 "$src/bin/saetta" "$PREFIX/bin/saetta"
say "installed $("$PREFIX/bin/saetta" version) into $PREFIX/bin"

case ":$PATH:" in *":$PREFIX/bin:"*) ;; *)
  say "add it to your PATH:  echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> ~/.zshrc && exec zsh" ;;
esac
cat <<MSG

Next:
  saetta download coder           # 76 GB into ~/Models (or: full, 105 GB)
  SAETTA_CTX=204800 saetta serve ~/Models/mbrown-uno/Qwen3.8-Flash-Next-GSQ-RCO-Coder-MLX
Docs: https://github.com/$REPO
MSG
