# Copyright 2026 TGS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import unittest

import numpy as np

import mdio_cpp as mdio

SCHEMA = {
    "metadata": {
        "apiVersion": "1.0.0",
        "name": "Demo MDIO",
        "createdOn": "2024-08-01T15:50:00.000000Z",
    },
    "variables": [
        {
            "name": "X",
            "dataType": "uint32",
            "dimensions": [{"name": "X", "size": 10}],
        },
        {
            "name": "Y",
            "dataType": "uint32",
            "dimensions": [{"name": "Y", "size": 10}],
        },
        {
            "name": "Grid",
            "dataType": "float32",
            "dimensions": ["X", "Y"],
        },
    ],
}


def _write_demo(path):
    ds = mdio.Dataset.from_json(SCHEMA, path, mode=mdio.OpenMode.CREATE)
    x = np.arange(10, dtype=np.uint32)
    y = np.arange(10, dtype=np.uint32)
    grid = np.arange(100, dtype=np.float32).reshape(10, 10)
    ds.variables["X"].write(x)
    ds.variables["Y"].write(y)
    ds.variables["Grid"].write(grid)
    return ds, x, y, grid


class TestMdioCppBindings(unittest.TestCase):
    def test_create_write_read_isel(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = tmp + "/demo.mdio"
            ds, x, y, grid = _write_demo(path)
            self.assertIn("Grid", ds.variables)
            self.assertEqual(ds.variables["Grid"].dtype_name, "float32")
            self.assertEqual(list(ds.variables["Grid"].shape), [10, 10])
            self.assertEqual(sorted(ds.variables.keys()), ["Grid", "X", "Y"])
            self.assertEqual(len(ds.variables), 3)
            self.assertEqual(set(ds.variables), {"Grid", "X", "Y"})

            reopened = mdio.Dataset.open(path)
            np.testing.assert_array_equal(reopened.variables["X"].read(), x)
            np.testing.assert_array_equal(reopened.variables["Grid"].read(), grid)

            data = reopened.variables["Grid"].read_data()
            reopened.variables["Grid"].write(data)
            np.testing.assert_array_equal(reopened.variables["Grid"].read(), grid)

            sliced = reopened.isel(X=slice(0, 4), Y=slice(2, 6))
            np.testing.assert_array_equal(
                sliced.variables["Grid"].read(), grid[0:4, 2:6]
            )

            by_range = reopened.isel(mdio.Range("X", 0, 4), mdio.Range("Y", 2, 6))
            np.testing.assert_array_equal(
                by_range.variables["Grid"].read(), grid[0:4, 2:6]
            )

            indexed = reopened.isel(X=3)
            self.assertEqual(list(indexed.variables["Grid"].shape), [1, 10])
            np.testing.assert_array_equal(
                indexed.variables["Grid"].read(), grid[3:4, :]
            )

            attrs = dict(reopened.variables["X"].attributes)
            if "attributes" not in attrs:
                attrs["attributes"] = {}
            attrs["attributes"]["foo"] = "bar"
            reopened.variables["X"].update_attributes(attrs)
            reopened.commit_metadata()

            with self.assertRaises(mdio.MdioError):
                reopened.variables["X"].update_attributes(
                    attrs, histogram_dtype="nope"
                )

            mdio.delete_dataset(path)

    def test_sel_value_and_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = tmp + "/demo.mdio"
            _, x, _, grid = _write_demo(path)
            ds = mdio.Dataset.open(path)

            picked = ds.sel(X=5)
            self.assertEqual(list(picked.variables["Grid"].shape), [1, 10])
            np.testing.assert_array_equal(picked.variables["X"].read(), [5])
            np.testing.assert_array_equal(
                picked.variables["Grid"].read(), grid[5:6, :]
            )

            by_value = ds.sel(mdio.Value("X", 5))
            np.testing.assert_array_equal(by_value.variables["X"].read(), [5])

            # C++ / xarray sel ranges are inclusive of stop.
            ranged = ds.sel(X=slice(2, 6))
            np.testing.assert_array_equal(ranged.variables["X"].read(), x[2:7])
            np.testing.assert_array_equal(
                ranged.variables["Grid"].read(), grid[2:7, :]
            )

            with self.assertRaises(mdio.MdioError) as ctx:
                ds.sel(X=[2, 4, 6])
            self.assertIn("ListDescriptor", str(ctx.exception))

            self.assertFalse(hasattr(mdio, "List"))
            self.assertFalse(hasattr(mdio, "kOpen"))
            self.assertEqual(mdio.__version__, mdio._core.__version__)

            mdio.delete_dataset(path)

    def test_trim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = tmp + "/demo.mdio"
            _write_demo(path)
            mdio.trim_dataset(path, {"X": 5})
            trimmed = mdio.Dataset.open(path)
            self.assertEqual(list(trimmed.variables["X"].shape), [5])
            self.assertEqual(list(trimmed.variables["Grid"].shape), [5, 10])
            mdio.delete_dataset(path)


if __name__ == "__main__":
    unittest.main()
