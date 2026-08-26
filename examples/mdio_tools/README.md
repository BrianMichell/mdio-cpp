# MDIO sidecars (`mdio_load`, `mdio_optimize`, `mdio_labels`)

Stand-alone CLIs for WEBKNOSSOS. They consume an **installed** `mdio-cpp` via `find_package(mdio)` and link `mdio::monolith`.

That is the same pattern as [Consuming the installed package](../../USER_GUIDE.md#consuming-the-installed-package) in the user guide. The older `examples/seismic_reader` example did the same idea with the installer plus a hand-written `-l` list; this CMake is the current form.

Sources stay in `tools/` (`mdio_load.cc`, `mdio_optimize.cc`, `mdio_labels.cc`). This directory is only the CMake project.

## Prerequisites

- CMake 3.27 *or better*
- NASM on `PATH`
- git (only if this tree is not already inside an mdio-cpp checkout)

## Bootstrap

From this directory:

```bash
./bootstrap.sh
```

That installs mdio-cpp into `inst/` (`-DMDIO_BUILD_MONOLITHIC_SHARED=ON`) and builds the three binaries into `build/`. In-tree checkout uses `../..`; otherwise it clones `https://github.com/TGSAI/mdio-cpp.git`. Re-run skips the mdio install if `inst/` already has `libmdio_monolith.so`. `--force` rebuilds it. `--prefix DIR` / `--jobs N` override defaults.

## Manual install (same steps as the script)

From the `mdio-cpp` repo root:

```bash
cmake -S . -B build-monolith \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install" \
  -DMDIO_BUILD_MONOLITHIC_SHARED=ON \
  -DBUILD_TESTING=OFF
cmake --build build-monolith --target install -j"$(nproc)"
```

Then from this directory:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../../install"
cmake --build build -j"$(nproc)"
```

Binaries:

- `build/mdio_load` — TensorStore reader sidecar (Play talks TCP localhost)
- `build/mdio_optimize` — write `fast_mag_*` into an MDIO store
- `build/mdio_labels` — write a labels variable from an annotation dump

Optional: `cmake --install build --prefix /path/to/mdio/install` puts them in that prefix's `bin/`.

## Point WEBKNOSSOS at them

Env vars (highest priority):

```bash
export MDIO_LOAD_BINARY="$PWD/build/mdio_load"
export MDIO_OPTIMIZE_BINARY="$PWD/build/mdio_optimize"
export MDIO_LABELS_BINARY="$PWD/build/mdio_labels"
```

Or set `datastore.mdio.{load,optimize,labels}Binary` in `application.conf`. Combined-dev defaults expect this example's `build/` next to a sibling `mdio-cpp` checkout.
