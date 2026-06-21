#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_path="$repo_root/.build/docc-site"
repository_name="${GITHUB_REPOSITORY:-rl-swift}"
hosting_base_path="${repository_name##*/}"

usage() {
  cat <<'EOF'
Usage: scripts/build-docc-site.sh [--output-path PATH] [--hosting-base-path PATH]

Builds static DocC output for the RLSwift core, Isaac Sim, and MuJoCo targets.
The MuJoCo target requires MuJoCo headers and libraries to be visible through
pkg-config before this script runs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-path)
      output_path="$2"
      shift 2
      ;;
    --hosting-base-path)
      hosting_base_path="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$output_path" in
  /*) ;;
  *) output_path="$repo_root/$output_path" ;;
esac

cd "$repo_root"
rm -rf "$output_path"
mkdir -p "$output_path"
touch "$output_path/.nojekyll"

renderer_assets_path() {
  local runtime_resource_path
  runtime_resource_path="$(
    swiftc -print-target-info \
      | sed -n 's/.*"runtimeResourcePath"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
      | head -n 1
  )"

  if [[ -n "$runtime_resource_path" ]]; then
    local candidate="${runtime_resource_path%/lib/swift}/share/docc/render"
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  return 1
}

ensure_renderer_assets() {
  local target="$1"
  local destination="$output_path/$target"
  local renderer_assets="$2"

  for asset_dir in css js; do
    if ! find "$destination/$asset_dir" -maxdepth 1 -type f -name 'index.*' 2>/dev/null | grep -q .; then
      if [[ ! -d "$renderer_assets/$asset_dir" ]]; then
        printf 'DocC renderer assets missing: %s\n' "$renderer_assets/$asset_dir" >&2
        exit 1
      fi

      rm -rf "$destination/$asset_dir"
      cp -R "$renderer_assets/$asset_dir" "$destination/$asset_dir"
    fi

    if ! find "$destination/$asset_dir" -maxdepth 1 -type f -name 'index.*' | grep -q .; then
      printf 'DocC %s renderer assets were not written for %s\n' "$asset_dir" "$target" >&2
      exit 1
    fi
  done
}

if ! docc_renderer_assets="$(renderer_assets_path)"; then
  printf 'Could not locate Swift-DocC renderer assets from swiftc -print-target-info\n' >&2
  exit 1
fi

generate_target() {
  local target="$1"
  shift
  local destination="$output_path/$target"
  mkdir -p "$destination"
  swift package \
    "$@" \
    --allow-writing-to-directory "$destination" \
    generate-documentation \
    --target "$target" \
    --output-path "$destination" \
    --transform-for-static-hosting \
    --hosting-base-path "$hosting_base_path/$target"
  ensure_renderer_assets "$target" "$docc_renderer_assets"
}

generate_target "RLSwift"
generate_target "RLSwiftIsaacSim"
generate_target "RLSwiftMuJoCo" --disable-default-traits --traits MuJoCoBackend

cat > "$output_path/index.html" <<'EOF'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RLSwift Documentation</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }
    body {
      margin: 0;
      padding: 48px;
      max-width: 960px;
    }
    h1 {
      margin: 0 0 12px;
      font-size: 2.4rem;
    }
    p {
      max-width: 720px;
      color: #555;
    }
    @media (prefers-color-scheme: dark) {
      p { color: #bbb; }
    }
    ul {
      display: grid;
      gap: 12px;
      padding: 0;
      list-style: none;
      max-width: 680px;
    }
    a {
      display: block;
      padding: 16px 18px;
      border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
      border-radius: 8px;
      color: inherit;
      text-decoration: none;
    }
    a:hover {
      border-color: currentColor;
    }
    strong {
      display: block;
      margin-bottom: 4px;
    }
    span {
      color: #666;
    }
    @media (prefers-color-scheme: dark) {
      span { color: #aaa; }
    }
  </style>
</head>
<body>
  <h1>RLSwift Documentation</h1>
  <p>Generated DocC documentation for the core reinforcement learning package and simulator bridge products.</p>
  <ul>
    <li><a href="./RLSwift/documentation/rlswift/"><strong>RLSwift</strong><span>Core environments, agents, robot safety, deployment, and training workflows.</span></a></li>
    <li><a href="./RLSwiftIsaacSim/documentation/rlswiftisaacsim/"><strong>RLSwiftIsaacSim</strong><span>JSON/HTTP bridge guide and API reference for Isaac Sim and Isaac Lab sidecars.</span></a></li>
    <li><a href="./RLSwiftMuJoCo/documentation/rlswiftmujoco/"><strong>RLSwiftMuJoCo</strong><span>Native MuJoCo simulation guide and API reference for Swift rollout loops.</span></a></li>
  </ul>
</body>
</html>
EOF

printf 'DocC site written to %s\n' "$output_path"
