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

"""Python bindings for the MDIO-cpp library."""

from mdio_cpp._core import (
    CoordinateSelector,
    Dataset,
    HeaderVariable,
    HeaderVariableCollection,
    Interval,
    MdioError,
    OpenMode,
    Range,
    UserAttributes,
    Value,
    Variable,
    VariableCollection,
    VariableData,
    ZarrVersion,
    __version__,
    construct,
    delete_dataset,
    dtypes,
    from_variable,
    trim_dataset,
    units,
    validate_dataset,
)

__all__ = [
    "CoordinateSelector",
    "Dataset",
    "HeaderVariable",
    "HeaderVariableCollection",
    "Interval",
    "MdioError",
    "OpenMode",
    "Range",
    "UserAttributes",
    "Value",
    "Variable",
    "VariableCollection",
    "VariableData",
    "ZarrVersion",
    "__version__",
    "construct",
    "delete_dataset",
    "dtypes",
    "from_variable",
    "trim_dataset",
    "units",
    "validate_dataset",
]
