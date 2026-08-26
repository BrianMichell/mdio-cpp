#!/usr/bin/env bash
# bootstrap.sh
#
# Install mdio-cpp (monolithic shared lib) and build mdio_load / mdio_optimize /
# mdio_labels against it with find_package(mdio).
#
# Same idea as examples/seismic_reader/bootstrap.sh, but uses cmake install +
# mdio::monolith instead of mdio-cpp-installer and a hand-written -l list.
#
# Usage:
#   ./bootstrap.sh
#   ./bootstrap.sh --force          # rebuild mdio even if inst/ already exists
#   ./bootstrap.sh --prefix /opt/mdio
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
INST="${MDIO_PREFIX:-$ROOT/inst}"
BUILD_MDIO="$ROOT/build-mdio"
BUILD_TOOLS="$ROOT/build"
FORCE=0
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

usage() {
  cat <<EOF
Usage: $0 [--force] [--prefix DIR] [--jobs N]

  --force         Rebuild and reinstall mdio-cpp even if inst/ is present.
  --prefix DIR    Install prefix (default: $ROOT/inst, or \$MDIO_PREFIX).
  --jobs N        Parallel build jobs (default: $JOBS).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --prefix)
      INST="${2:?--prefix needs a directory}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:?--jobs needs a number}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required tool: $1" >&2
    exit 1
  }
}

need cmake
if ! command -v nasm >/dev/null 2>&1; then
  echo "Missing required tool: nasm (mdio-cpp / Tensorstore need it on PATH)" >&2
  exit 1
fi

CMAKE_VER="$(cmake --version | head -n1 | awk '{print $3}')"
if ! printf '%s\n%s\n' "3.27" "$CMAKE_VER" | sort -C -V; then
  echo "CMake 3.27 or better required (found ${CMAKE_VER})" >&2
  exit 1
fi

if [[ -f "$ROOT/../../CMakeLists.txt" && -d "$ROOT/../../mdio" ]]; then
  MDIO_SRC="$(cd "$ROOT/../.." && pwd)"
  echo "Using in-tree mdio-cpp: $MDIO_SRC"
else
  need git
  MDIO_SRC="$ROOT/mdio-cpp"
  if [[ ! -d "$MDIO_SRC/.git" ]]; then
    echo "Cloning mdio-cpp into $MDIO_SRC"
    git clone https://github.com/TGSAI/mdio-cpp.git "$MDIO_SRC"
  else
    echo "Using cloned mdio-cpp: $MDIO_SRC"
  fi
fi

have_install=0
if [[ "$FORCE" -eq 0 ]]; then
  for cand in \
      "$INST/lib/libmdio_monolith.so" \
      "$INST/lib64/libmdio_monolith.so"; do
    [[ -f "$cand" ]] || continue
    for cfg in \
        "$INST/lib/cmake/mdio/mdioConfig.cmake" \
        "$INST/lib64/cmake/mdio/mdioConfig.cmake"; do
      if [[ -f "$cfg" ]]; then
        have_install=1
        break 2
      fi
    done
  done
fi

if [[ "$have_install" -eq 1 ]]; then
  echo "Reusing mdio install at $INST"
else
  echo "Installing mdio-cpp (monolith) into $INST"
  cmake -S "$MDIO_SRC" -B "$BUILD_MDIO" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INST" \
    -DMDIO_BUILD_MONOLITHIC_SHARED=ON \
    -DBUILD_TESTING=OFF
  cmake --build "$BUILD_MDIO" --target install -j"$JOBS"
fi

echo "Building sidecars against $INST"
cmake -S "$ROOT" -B "$BUILD_TOOLS" -DCMAKE_PREFIX_PATH="$INST"
cmake --build "$BUILD_TOOLS" -j"$JOBS"

echo
echo "Done. Binaries:"
echo "  $BUILD_TOOLS/mdio_load"
echo "  $BUILD_TOOLS/mdio_optimize"
echo "  $BUILD_TOOLS/mdio_labels"
echo
echo "WEBKNOSSOS:"
echo "  export MDIO_LOAD_BINARY=\"$BUILD_TOOLS/mdio_load\""
echo "  export MDIO_OPTIMIZE_BINARY=\"$BUILD_TOOLS/mdio_optimize\""
echo "  export MDIO_LABELS_BINARY=\"$BUILD_TOOLS/mdio_labels\""
echo
echo "libmdio_monolith.so is in $INST/lib (RPATH is already set on the binaries)."
