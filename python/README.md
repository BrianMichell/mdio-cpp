# MDIO-cpp Python bindings

pybind11 wrapper around the C++ MDIO API. Import as `mdio_cpp` so it does not
clash with the standalone `mdio` Python package.

## Build

From the repo root, reuse an existing CMake build tree:

```bash
cmake -S . -B build -DMDIO_BUILD_PYTHON=ON
cmake --build build --target _core -j"$(nproc)"
PYTHONPATH=python/src python3 -c "import mdio_cpp; print(mdio_cpp.Dataset)"
```

Needs Python 3.9+, NumPy, and the same C++ toolchain as MDIO-cpp.

## Usage

```python
import numpy as np
import mdio_cpp as mdio

schema = {
    "metadata": {
        "apiVersion": "1.0.0",
        "name": "Demo MDIO",
        "createdOn": "2024-08-01T15:50:00.000000Z",
    },
    "variables": [
        {"name": "X", "dataType": "uint32",
         "dimensions": [{"name": "X", "size": 10}]},
        {"name": "Y", "dataType": "uint32",
         "dimensions": [{"name": "Y", "size": 10}]},
        {"name": "Grid", "dataType": "float32", "dimensions": ["X", "Y"]},
    ],
}

ds = mdio.Dataset.from_json(schema, "demo.mdio", mode=mdio.OpenMode.CREATE)
ds.variables["Grid"].write(np.arange(100, dtype=np.float32).reshape(10, 10))

ds = mdio.Dataset.open("demo.mdio")
arr = ds.variables["Grid"].read()          # numpy.ndarray
sliced = ds.isel(X=slice(0, 4), Y=slice(2, 6))
ds.variables["X"].update_attributes({"attributes": {"foo": "bar"}})
ds.commit_metadata()
```

C++ `Result` / `Future` errors become `mdio_cpp.MdioError`. Reads and writes
block, then return values.

`isel` integer kwargs keep a size-1 dimension (they do not drop it the way
xarray does). `sel` takes a scalar value or a label range (`slice` / `Range` /
`(start, stop)`). Python lists are list-sel and not implemented. `sel` range
bounds are inclusive, matching C++ / xarray.

`trim_dataset` accepts `{label: new_stop}`, `Range`, or `slice`. Only the stop
is applied on disk.

Open with `OpenMode` or the strings `"r"` / `"w-"` / `"w"`.

## Covered API

- `Dataset`: `from_json`, `open`, `isel`, `sel`, `select_field`,
  `commit_metadata`, `intervals`, `variables`, `header_variables`,
  `coordinates`, `domain`, `metadata`, `ds["name"]`
- `Variable` / `VariableData`: read/write NumPy, slice, attributes, spec,
  chunk/store shape, units
- `VariableCollection`, `HeaderVariable`, `HeaderVariableCollection`
- `Range` / `Value` descriptors
- `UserAttributes`, `CoordinateSelector`
- `validate_dataset`, `construct`, `delete_dataset`, `trim_dataset`
- `OpenMode`, `ZarrVersion`, `units`, `dtypes`
