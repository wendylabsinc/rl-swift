#!/usr/bin/env bash
set -euo pipefail

version="${MUJOCO_VERSION:-3.9.0}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_root="${MUJOCO_SDK_ROOT:-"$repo_root/.build/mujoco-sdk"}"
downloads="$sdk_root/downloads"
system_name="$(uname -s)"
machine="$(uname -m)"

case "$system_name:$machine" in
  Linux:x86_64)
    platform="linux-x86_64"
    asset="mujoco-$version-$platform.tar.gz"
    ;;
  Linux:aarch64|Linux:arm64)
    platform="linux-aarch64"
    asset="mujoco-$version-$platform.tar.gz"
    ;;
  Darwin:*)
    platform="macos-universal2"
    asset="mujoco-$version-$platform.dmg"
    ;;
  *)
    printf 'Unsupported MuJoCo SDK platform: %s %s\n' "$system_name" "$machine" >&2
    exit 1
    ;;
esac

url="https://github.com/google-deepmind/mujoco/releases/download/$version/$asset"
sha_url="$url.sha256"
archive="$downloads/$asset"
sha_file="$downloads/$asset.sha256"

mkdir -p "$downloads" "$sdk_root/include" "$sdk_root/lib/pkgconfig"

if [[ ! -f "$archive" ]]; then
  printf 'Downloading MuJoCo %s for %s\n' "$version" "$platform" >&2
  curl -fL --retry 3 "$url" -o "$archive"
fi

if [[ ! -f "$sha_file" ]]; then
  curl -fsSL "$sha_url" -o "$sha_file"
fi

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$downloads" && sha256sum -c "$(basename "$sha_file")") >&2
elif command -v shasum >/dev/null 2>&1; then
  (cd "$downloads" && shasum -a 256 -c "$(basename "$sha_file")") >&2
fi

if [[ "$system_name" == "Darwin" ]]; then
  mount_output="$(hdiutil attach -nobrowse -readonly "$archive")"
  volume="$(printf '%s\n' "$mount_output" | awk '/\/Volumes\// {print $3; exit}')"
  if [[ -z "$volume" ]]; then
    printf 'Could not locate mounted MuJoCo volume\n' >&2
    printf '%s\n' "$mount_output" >&2
    exit 1
  fi
  cleanup() {
    hdiutil detach "$volume" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT

  framework="$volume/mujoco.framework/Versions/A"
  rm -rf "$sdk_root/include/mujoco" "$sdk_root/lib/mujoco.framework"
  mkdir -p "$sdk_root/include/mujoco" "$sdk_root/lib/mujoco.framework/Versions/A"
  cp -R "$framework/Headers/"* "$sdk_root/include/mujoco/"
  cp "$framework/libmujoco.$version.dylib" "$sdk_root/lib/"
  cp "$framework/libmujoco.$version.dylib" "$sdk_root/lib/mujoco.framework/Versions/A/"
  ln -sf "libmujoco.$version.dylib" "$sdk_root/lib/libmujoco.dylib"

  for build_dir in "$repo_root/.build/$machine-apple-macosx/debug" "$repo_root/.build/debug"; do
    mkdir -p "$build_dir/mujoco.framework/Versions/A"
    cp "$framework/libmujoco.$version.dylib" "$build_dir/mujoco.framework/Versions/A/"
  done
else
  extract_dir="$downloads/extracted"
  rm -rf "$extract_dir"
  mkdir -p "$extract_dir"
  tar -xzf "$archive" -C "$extract_dir"
  source_dir="$extract_dir/mujoco-$version"
  rm -rf "$sdk_root/include/mujoco"
  mkdir -p "$sdk_root/include" "$sdk_root/lib"
  cp -R "$source_dir/include/mujoco" "$sdk_root/include/"
  cp "$source_dir/lib/libmujoco.so.$version" "$sdk_root/lib/"
  cp "$source_dir/lib/libmujoco.so" "$sdk_root/lib/"
fi

cat > "$sdk_root/lib/pkgconfig/mujoco.pc" <<EOF
prefix=$sdk_root
includedir=\${prefix}/include
libdir=\${prefix}/lib

Name: mujoco
Description: MuJoCo physics engine
Version: $version
Cflags: -I\${includedir}
Libs: -L\${libdir} -lmujoco
EOF

pkg_config_path="$sdk_root/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
library_path="$sdk_root/lib"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    printf 'MUJOCO_SDK_ROOT=%s\n' "$sdk_root"
    printf 'PKG_CONFIG_PATH=%s\n' "$pkg_config_path"
    printf 'LD_LIBRARY_PATH=%s%s\n' "$library_path" "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    printf 'DYLD_LIBRARY_PATH=%s%s\n' "$library_path" "${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  } >> "$GITHUB_ENV"
fi

printf 'export MUJOCO_SDK_ROOT=%q\n' "$sdk_root"
printf 'export PKG_CONFIG_PATH=%q\n' "$pkg_config_path"
printf 'export LD_LIBRARY_PATH=%q\n' "$library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf 'export DYLD_LIBRARY_PATH=%q\n' "$library_path${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
