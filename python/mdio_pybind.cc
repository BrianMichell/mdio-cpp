#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "nlohmann/json.hpp"
#include "mdio/dataset.h"
#include "mdio/dataset_factory.h"
#include "mdio/impl.h"
#include "mdio/variable.h"

namespace py = pybind11;

using mdio::Dataset;
using mdio::DimensionIdentifier;
using mdio::Index;
using mdio::RangeDescriptor;
using mdio::Result;
using mdio::Variable;
using mdio::VariableData;
using mdio::ValueDescriptor;
using mdio::ListDescriptor;

namespace {

[[noreturn]] void ThrowStatus(const absl::Status& status) {
  throw py::value_error(status.ToString());
}

template <typename T>
T Unwrap(const Result<T>& result) {
  if (!result.ok()) {
    ThrowStatus(result.status());
  }
  return result.value();
}

void Wait(const mdio::Future<void>& fut) {
  auto res = fut.result();
  if (!res.ok()) {
    ThrowStatus(res.status());
  }
}

template <typename T>
T Wait(const mdio::Future<T>& fut) {
  auto res = fut.result();
  return Unwrap(res);
}

// Utility to convert objects with an ostream operator into a string.
template <typename T>
std::string StreamToString(const T& value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

py::object JsonToPy(const nlohmann::json& json) {
  auto json_mod = py::module_::import("json");
  auto dumped = json.dump();
  return json_mod.attr("loads")(py::str(dumped));
}

nlohmann::json PyToJson(const py::handle& handle) {
  auto json_mod = py::module_::import("json");
  auto dumped = json_mod.attr("dumps")(handle).cast<std::string>();
  return nlohmann::json::parse(dumped);
}

nlohmann::json NormalizeSpecForPickle(nlohmann::json spec) {
  if (spec.contains("kvstore") && spec["kvstore"].contains("path")) {
    std::string path = spec["kvstore"]["path"];
    if (!path.empty() && path.back() == '/') {
      path.pop_back();
      spec["kvstore"]["path"] = path;
    }
  }

  const bool is_byte = spec.contains("dtype") && spec["dtype"] == "byte";

  // Drop stale constraints that can cause rank/dtype mismatches on reopen.
  spec.erase("schema");
  spec.erase("transform");
  spec.erase("dtype");  // let metadata drive dtype

  if (is_byte) {
    spec.erase("field");          // avoid selecting a single field
    spec["open_as_void"] = true;  // request raw-byte view on reopen
  }

  return spec;
}

py::dict DomainToDict(const tensorstore::IndexDomain<>& domain) {
  py::dict out;
  std::vector<Index> origin(domain.rank());
  std::vector<Index> shape(domain.rank());
  std::vector<std::string> labels(domain.rank());
  for (tensorstore::DimensionIndex i = 0; i < domain.rank(); ++i) {
    origin[i] = domain[i].interval().inclusive_min();
    shape[i] = domain[i].interval().size();
    labels[i] = std::string(domain.labels()[i]);
  }
  out["rank"] = domain.rank();
  out["origin"] = origin;
  out["shape"] = shape;
  out["labels"] = labels;
  return out;
}

py::dtype DataTypeToDtype(const mdio::DataType& dtype) {
  using mdio::constants::kBool;
  using mdio::constants::kByte;
  using mdio::constants::kComplex128;
  using mdio::constants::kComplex64;
  using mdio::constants::kFloat16;
  using mdio::constants::kFloat32;
  using mdio::constants::kFloat64;
  using mdio::constants::kInt16;
  using mdio::constants::kInt32;
  using mdio::constants::kInt64;
  using mdio::constants::kInt8;
  using mdio::constants::kUint16;
  using mdio::constants::kUint32;
  using mdio::constants::kUint64;
  using mdio::constants::kUint8;

  if (dtype == kBool) return py::dtype::of<bool>();
  if (dtype == kByte) return py::dtype::of<std::byte>();
  if (dtype == kInt8) return py::dtype::of<int8_t>();
  if (dtype == kInt16) return py::dtype::of<int16_t>();
  if (dtype == kInt32) return py::dtype::of<int32_t>();
  if (dtype == kInt64) return py::dtype::of<int64_t>();
  if (dtype == kUint8) return py::dtype::of<uint8_t>();
  if (dtype == kUint16) return py::dtype::of<uint16_t>();
  if (dtype == kUint32) return py::dtype::of<uint32_t>();
  if (dtype == kUint64) return py::dtype::of<uint64_t>();
  if (dtype == kFloat16) return py::dtype("float16");
  if (dtype == kFloat32) return py::dtype::of<float>();
  if (dtype == kFloat64) return py::dtype::of<double>();
  if (dtype == kComplex64) return py::dtype::of<std::complex<float>>();
  if (dtype == kComplex128) return py::dtype::of<std::complex<double>>();

  // Fallback to Python object dtype
  return py::dtype("object");
}

template <typename Array>
py::array SharedArrayToNumpy(const Array& array) {
  auto dtype = DataTypeToDtype(array.dtype());
  std::vector<ssize_t> shape;
  std::vector<ssize_t> strides;

  for (auto dim : array.shape()) {
    shape.push_back(static_cast<ssize_t>(dim));
  }
  for (auto stride : array.byte_strides()) {
    strides.push_back(static_cast<ssize_t>(stride));
  }

  using ArrayType = std::decay_t<decltype(array)>;
  auto holder = new ArrayType(array);
  py::capsule base(holder, [](void* ptr) {
    delete static_cast<ArrayType*>(ptr);
  });

  return py::array(dtype, shape, strides,
                   array.byte_strided_origin_pointer().get(), base);
}

py::dict VariableDataToDict(
    VariableData<void, mdio::dynamic_rank, mdio::offset_origin> data) {
  py::dict out;
  out["metadata"] = JsonToPy(data.metadata);
  out["domain"] = DomainToDict(data.dimensions());
  out["data"] = SharedArrayToNumpy(data.get_data_accessor());
  return out;
}

// Wrapper to keep VariableData alive in Python while exposing NumPy views.
struct PyVariableData {
  VariableData<void, mdio::dynamic_rank, mdio::offset_origin> data;

  py::dict to_dict() const { return VariableDataToDict(data); }
};

struct PyWriteFutures {
  mdio::WriteFutures futures;

  void wait_copy() const { Wait(futures.copy_future); }
  void wait_commit() const { Wait(futures.commit_future); }
  void wait_all() const {
    wait_copy();
    wait_commit();
  }
};

RangeDescriptor<Index> MakeRange(std::string label, Index start,
                                 Index stop, Index step) {
  RangeDescriptor<Index> desc;
  // DimensionIdentifier stores a non-owning view of the label string.  Intern
  // the label so the backing storage remains alive for the lifetime of any
  // RangeDescriptor returned to Python.
  const std::string& interned_label = [&label]() -> const std::string& {
    static std::mutex mu;
    static std::set<std::string> labels;
    std::lock_guard<std::mutex> lock(mu);
    auto [it, inserted] = labels.insert(std::move(label));
    return *it;
  }();

  desc.label = DimensionIdentifier(std::string_view(interned_label));
  desc.start = start;
  desc.stop = stop;
  desc.step = step;
  return desc;
}

RangeDescriptor<Index> SliceToRange(std::string label,
                                    const py::slice& s) {
  py::object py_start = s.attr("start");
  py::object py_stop = s.attr("stop");
  py::object py_step = s.attr("step");
  if (py_start.is_none() || py_stop.is_none()) {
    throw py::value_error("Slice start/stop must be set when using isel");
  }
  Index start = py_start.cast<Index>();
  Index stop = py_stop.cast<Index>();
  Index step = py_step.is_none() ? 1 : py_step.cast<Index>();
  return MakeRange(std::move(label), start, stop, step);
}

std::vector<RangeDescriptor<Index>> ParseRangeList(const py::list& ranges) {
  std::vector<RangeDescriptor<Index>> out;
  out.reserve(ranges.size());
  for (const auto& item : ranges) {
    if (py::isinstance<RangeDescriptor<Index>>(item)) {
      out.push_back(item.cast<RangeDescriptor<Index>>());
      continue;
    }
    if (py::isinstance<py::tuple>(item)) {
      auto tpl = item.cast<py::tuple>();
      if (tpl.size() == 2 && py::isinstance<py::slice>(tpl[1])) {
        auto label = tpl[0].cast<std::string>();
        out.emplace_back(SliceToRange(std::move(label), tpl[1].cast<py::slice>()));
        continue;
      }
      if (tpl.size() < 3 || tpl.size() > 4) {
        throw py::value_error(
            "Slice tuple must be (label, start, stop[, step]) or (label, slice)");
      }
      auto label = tpl[0].cast<std::string>();
      auto start = tpl[1].cast<Index>();
      auto stop = tpl[2].cast<Index>();
      Index step = 1;
      if (tpl.size() == 4) {
        step = tpl[3].cast<Index>();
      }
      out.emplace_back(MakeRange(std::move(label), start, stop, step));
      continue;
    }
    throw py::value_error(
        "Slices must be tuples (label, start, stop[, step]) or (label, slice) "
        "or RangeDescriptor");
  }
  return out;
}

auto ToOpenMode(const std::string& mode) {
  if (mode == "open") {
    return mdio::constants::kOpen;
  } else if (mode == "create") {
    return mdio::constants::kCreate;
  } else if (mode == "create_clean") {
    return mdio::constants::kCreateClean;
  } else {
    throw py::value_error("Invalid open_mode. Use 'open', 'create', or "
                          "'create_clean'.");
  }
}

Dataset OpenDataset(const std::string& path, const std::string& mode) {
  auto open_mode = ToOpenMode(mode);
  auto fut = Dataset::Open(path, open_mode);
  return Wait(fut);
}

Dataset DatasetFromJson(const py::handle& schema_obj, const std::string& path,
                        const std::string& mode) {
  nlohmann::json schema = PyToJson(schema_obj);
  auto open_mode = ToOpenMode(mode);
  auto fut = Dataset::from_json(schema, path, open_mode);
  return Wait(fut);
}

Dataset DatasetFromSpecs(const py::handle& metadata_obj,
                         const py::list& variables_list,
                         const std::string& mode) {
  auto open_mode = ToOpenMode(mode);
  nlohmann::json meta = PyToJson(metadata_obj);
  std::vector<nlohmann::json> vars;
  vars.reserve(py::len(variables_list));
  for (const auto& v : variables_list) {
    vars.push_back(PyToJson(v));
  }
  auto fut = Dataset::Open(meta, vars, open_mode);
  return Wait(fut);
}

Variable<> SelectField(Dataset& dataset, const std::string& variable_name,
                       const std::string& field) {
  auto fut = dataset.SelectField(variable_name, field);
  return Wait(fut);
}

enum class PyValKind { kInt, kFloat, kString };
enum class PyDescKind { kValue, kList };

PyValKind DetectPyValKind(const py::handle& obj) {
  if (py::isinstance<py::int_>(obj)) return PyValKind::kInt;
  if (py::isinstance<py::float_>(obj)) return PyValKind::kFloat;
  if (py::isinstance<py::str>(obj)) return PyValKind::kString;
  throw py::value_error("Unsupported selector value type. Use int, float, or str.");
}

template <typename T>
T CastVal(const py::handle& obj);

template <>
int64_t CastVal<int64_t>(const py::handle& obj) {
  return obj.cast<int64_t>();
}

template <>
double CastVal<double>(const py::handle& obj) {
  return obj.cast<double>();
}

template <>
std::string CastVal<std::string>(const py::handle& obj) {
  return obj.cast<std::string>();
}

template <typename T>
std::vector<ValueDescriptor<T>> BuildValueDescriptors(const py::list& selectors) {
  std::vector<ValueDescriptor<T>> out;
  out.reserve(selectors.size());
  for (const auto& item : selectors) {
    if (!py::isinstance<py::tuple>(item)) {
      throw py::value_error("Value selector must be tuple (label, value)");
    }
    auto tpl = item.cast<py::tuple>();
    if (tpl.size() != 2) {
      throw py::value_error("Value selector tuple must be (label, value)");
    }
    auto label = tpl[0].cast<std::string>();
    auto value = CastVal<T>(tpl[1]);
    out.push_back(ValueDescriptor<T>{DimensionIdentifier(std::move(label)), value});
  }
  return out;
}

template <typename T>
std::vector<ListDescriptor<T>> BuildListDescriptors(const py::list& selectors) {
  std::vector<ListDescriptor<T>> out;
  out.reserve(selectors.size());
  for (const auto& item : selectors) {
    if (!py::isinstance<py::tuple>(item)) {
      throw py::value_error("List selector must be tuple (label, [values])");
    }
    auto tpl = item.cast<py::tuple>();
    if (tpl.size() != 2 || !py::isinstance<py::list>(tpl[1])) {
      throw py::value_error("List selector must be (label, list_of_values)");
    }
    auto label = tpl[0].cast<std::string>();
    auto py_vals = tpl[1].cast<py::list>();
    std::vector<T> values;
    values.reserve(py_vals.size());
    for (const auto& v : py_vals) {
      values.push_back(CastVal<T>(v));
    }
    out.push_back(ListDescriptor<T>{DimensionIdentifier(std::move(label)), values});
  }
  return out;
}

template <typename Descriptor>
Result<Dataset> SelDispatch(Dataset& ds, const std::vector<Descriptor>& v) {
  switch (v.size()) {
    case 1:
      return ds.sel(v[0]);
    case 2:
      return ds.sel(v[0], v[1]);
    case 3:
      return ds.sel(v[0], v[1], v[2]);
    case 4:
      return ds.sel(v[0], v[1], v[2], v[3]);
    case 5:
      return ds.sel(v[0], v[1], v[2], v[3], v[4]);
    case 6:
      return ds.sel(v[0], v[1], v[2], v[3], v[4], v[5]);
    case 7:
      return ds.sel(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
    case 8:
      return ds.sel(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    default:
      return absl::InvalidArgumentError(
          "Too many selectors for sel; maximum supported is 8.");
  }
}

Dataset DatasetSel(Dataset& ds, const py::list& selectors) {
  if (selectors.empty()) {
    throw py::value_error("sel requires at least one selector");
  }
  // Expect tuples. Determine descriptor kind and value kind from first entry.
  if (!py::isinstance<py::tuple>(selectors[0])) {
    throw py::value_error("Selectors must be tuples");
  }
  auto first_tpl = selectors[0].cast<py::tuple>();
  if (first_tpl.size() != 2) {
    throw py::value_error("Selector must be (label, value) or (label, list)");
  }
  bool is_list = py::isinstance<py::list>(first_tpl[1]);
  PyValKind kind = is_list ? DetectPyValKind(first_tpl[1].cast<py::list>()[0])
                           : DetectPyValKind(first_tpl[1]);

  // Ensure consistency
  for (const auto& item : selectors) {
    if (!py::isinstance<py::tuple>(item)) {
      throw py::value_error("Selectors must be tuples");
    }
    auto tpl = item.cast<py::tuple>();
    if (tpl.size() != 2) {
      throw py::value_error("Selector must be (label, value) or (label, list)");
    }
    bool this_is_list = py::isinstance<py::list>(tpl[1]);
    if (this_is_list != is_list) {
      throw py::value_error(
          "All selectors must be value selectors or list selectors, not mixed.");
    }
    if (is_list) {
      auto lst = tpl[1].cast<py::list>();
      if (lst.empty()) {
        throw py::value_error("List selector values cannot be empty");
      }
      auto k = DetectPyValKind(lst[0]);
      if (k != kind) {
        throw py::value_error("All selector values must share the same type");
      }
    } else {
      auto k = DetectPyValKind(tpl[1]);
      if (k != kind) {
        throw py::value_error("All selector values must share the same type");
      }
    }
  }

  Result<Dataset> res;
  if (is_list) {
    if (kind == PyValKind::kInt) {
      auto descs = BuildListDescriptors<int64_t>(selectors);
      res = SelDispatch(ds, descs);
    } else if (kind == PyValKind::kFloat) {
      auto descs = BuildListDescriptors<double>(selectors);
      res = SelDispatch(ds, descs);
    } else {
      auto descs = BuildListDescriptors<std::string>(selectors);
      res = SelDispatch(ds, descs);
    }
  } else {
    if (kind == PyValKind::kInt) {
      auto descs = BuildValueDescriptors<int64_t>(selectors);
      res = SelDispatch(ds, descs);
    } else if (kind == PyValKind::kFloat) {
      auto descs = BuildValueDescriptors<double>(selectors);
      res = SelDispatch(ds, descs);
    } else {
      auto descs = BuildValueDescriptors<std::string>(selectors);
      res = SelDispatch(ds, descs);
    }
  }
  return Unwrap(res);
}

Dataset DatasetIselPy(Dataset& ds, const py::list& slices) {
  auto descs = ParseRangeList(slices);
  if (descs.empty()) {
    throw py::value_error("isel requires at least one slice");
  }

  mdio::VariableCollection vars;
  std::map<std::string, tensorstore::IndexDomainDimension<>> dims;
  std::vector<std::string> keys = ds.variables.get_iterable_accessor();

  for (const auto& name : keys) {
    auto var_res = ds.variables.at(name);
    if (!var_res.ok()) {
      ThrowStatus(var_res.status());
    }
    // Apply all slices at once to the variable
    py::object py_var = py::cast(var_res.value());
    py_var = py_var.attr("slice")(slices);
    auto sliced = py_var.cast<Variable<>>();
    vars.add(name, sliced);

    mdio::DimensionIndex idx = 0;
    for (const auto label : sliced.get_store().domain().labels()) {
      if (!label.empty()) {
        dims[label] = sliced.get_store().domain()[idx];
      }
      ++idx;
    }
  }

  size_t size = dims.size();
  std::vector<std::string> labels(size);
  std::vector<Index> origin(size);
  std::vector<Index> shape(size);

  mdio::DimensionIndex idx = 0;
  for (const auto& [key, val] : dims) {
    labels[idx] = key;
    origin[idx] = val.interval().inclusive_min();
    shape[idx] = val.interval().size();
    ++idx;
  }

  auto domain_builder = tensorstore::IndexDomainBuilder<>(size)
                            .origin(origin)
                            .shape(shape)
                            .labels(labels);
  auto new_domain_res = domain_builder.Finalize();
  if (!new_domain_res.ok()) {
    ThrowStatus(new_domain_res.status());
  }

  return Dataset{ds.getMetadata(), vars, ds.coordinates, new_domain_res.value()};
}

PyVariableData ReadVariableData(Variable<>& var) {
  auto fut = var.Read();
  auto data =
      Wait<VariableData<void, mdio::dynamic_rank, mdio::offset_origin>>(fut);
  return PyVariableData{std::move(data)};
}

PyVariableData AllocateVariableData(Variable<>& var) {
  auto res = mdio::from_variable(var);
  return PyVariableData{Unwrap(res)};
}

void WriteVariableData(Variable<>& var, const PyVariableData& value) {
  auto futures = var.Write(value.data);
  Wait(futures.copy_future);
  Wait(futures.commit_future);
}

PyWriteFutures WriteVariableDataAsync(Variable<>& var,
                                      const PyVariableData& value) {
  auto futures = var.Write(value.data);
  return PyWriteFutures{std::move(futures)};
}

Variable<> UnpickleVariable(const py::object& json_spec_obj) {
  auto json_spec = PyToJson(json_spec_obj);
  auto var_future = Variable<>::Open(json_spec);
  return Wait(var_future);
}

Dataset UnpickleDataset(const py::object& metadata_obj,
                        const py::list& variables_list) {
  return DatasetFromSpecs(metadata_obj, variables_list, "open");
}

py::list IntervalsToPy(const std::vector<Variable<>::Interval>& ivals) {
  py::list out;
  for (const auto& iv : ivals) {
    py::dict d;
    d["label"] = std::string(iv.label.label());
    d["inclusive_min"] = iv.inclusive_min;
    d["exclusive_max"] = iv.exclusive_max;
    out.append(std::move(d));
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(mdio_cpp, m) {
  m.doc() = "Pybind11 bindings for the MDIO C++ library";

  py::class_<RangeDescriptor<Index>>(m, "RangeDescriptor")
      .def(py::init(&MakeRange), py::arg("label"), py::arg("start"),
           py::arg("stop"), py::arg("step") = 1)
      .def_readwrite("label", &RangeDescriptor<Index>::label)
      .def_readwrite("start", &RangeDescriptor<Index>::start)
      .def_readwrite("stop", &RangeDescriptor<Index>::stop)
      .def_readwrite("step", &RangeDescriptor<Index>::step)
      .def("__repr__", [](const RangeDescriptor<Index>& r) {
        return py::str("RangeDescriptor({}, {}, {}, {})")
            .format(std::string(r.label.label()), r.start, r.stop, r.step);
      });

  py::class_<Variable<>>(m, "Variable")
      .def(py::init([](const py::object& json_spec_obj) {
        auto json_spec = PyToJson(json_spec_obj);
        auto var_future = Variable<>::Open(json_spec);
        return Wait(var_future);
      }), py::arg("json_spec"))
      .def("read", [](Variable<>& self) {
        auto fut = self.Read();
        auto data =
            Wait<VariableData<void, mdio::dynamic_rank, mdio::offset_origin>>(
                fut);
        return VariableDataToDict(data);
      })
      .def("slice",
           [](Variable<>& self, const py::list& slices) {
             auto descs = ParseRangeList(slices);
             auto res = self.slice(descs);
             return Unwrap(res);
           },
           py::arg("slices"))
      .def("get_spec",
           [](const Variable<>& self) { return JsonToPy(Unwrap(self.get_spec())); })
      .def("get_metadata",
           [](const Variable<>& self) { return JsonToPy(self.getMetadata()); })
      .def("get_intervals",
           [](const Variable<>& self) {
             return IntervalsToPy(Unwrap(self.get_intervals()));
           })
      .def("get_attributes",
           [](const Variable<>& self) { return JsonToPy(self.GetAttributes()); })
      .def("update_attributes",
           [](Variable<>& self, const py::handle& attrs) {
             auto j = PyToJson(attrs);
             auto res = self.UpdateAttributes(j);
             if (!res.ok()) ThrowStatus(res.status());
           },
           py::arg("attributes"))
      .def("get_units", [](const Variable<>& self) {
        return JsonToPy(Unwrap(self.get_units()));
      })
      .def("read_data", &ReadVariableData,
           "Read variable into a VariableData handle that exposes a NumPy view.")
      .def("allocate_data", &AllocateVariableData,
           "Allocate an in-memory VariableData with default fill values.")
      .def("write_data", &WriteVariableData, py::arg("variable_data"),
           "Write a VariableData handle back to the variable.")
      .def("write_data_async", &WriteVariableDataAsync, py::arg("variable_data"),
           "Asynchronously write VariableData; returns WriteFutures for waiting.")
      .def("publish_metadata",
           [](Variable<>& self) {
             auto fut = self.PublishMetadata();
             Wait(fut);
           })
      .def("was_updated", &Variable<>::was_updated)
      .def("should_publish", &Variable<>::should_publish)
      .def("set_metadata_publish_flag", &Variable<>::set_metadata_publish_flag,
           py::arg("should_publish") = true)
      .def("get_chunk_shape", &Variable<>::get_chunk_shape)
      .def("get_store_shape", &Variable<>::get_store_shape)
      .def("num_samples", &Variable<>::num_samples)
      .def("rank", &Variable<>::rank)
      .def("dtype",
           [](const Variable<>& self) { return std::string(self.dtype().name()); })
      .def("dimensions", [](const Variable<>& self) {
        return DomainToDict(self.dimensions());
      })
      .def("__repr__", [](const Variable<>& self) {
        return StreamToString(self);
      })
      .def("__str__", [](const Variable<>& self) {
        return StreamToString(self);
      })
      .def_static("_open_from_spec", [](const py::object& json_spec_obj) {
        auto json_spec = PyToJson(json_spec_obj);
        auto var_future = Variable<>::Open(json_spec);
        return Wait(var_future);
      }, py::arg("json_spec"))
      .def("__reduce__", [](const py::object& self) {
        auto var = self.cast<const Variable<>&>();
        auto spec = NormalizeSpecForPickle(Unwrap(var.get_spec()));
        auto json_spec = JsonToPy(spec);
        return py::make_tuple(self.attr("__class__"), py::make_tuple(json_spec));
      });

  py::class_<Dataset>(m, "Dataset")
      .def(py::init([](const py::handle& metadata_obj,
                       const py::list& variables_list,
                       const std::string& mode) {
             return DatasetFromSpecs(metadata_obj, variables_list, mode);
           }),
           py::arg("metadata"), py::arg("variables"),
           py::arg("open_mode") = "open")
      .def_static("open", &OpenDataset, py::arg("path"),
                  py::arg("open_mode") = "open")
      .def_static("from_json", &DatasetFromJson, py::arg("schema"),
                  py::arg("path"), py::arg("open_mode") = "create")
      .def_static("from_specs", &DatasetFromSpecs, py::arg("metadata"),
                  py::arg("variables"), py::arg("open_mode") = "create")
      .def("isel",
           [](Dataset& self, const py::list& slices) {
             return DatasetIselPy(self, slices);
           },
           py::arg("slices"))
      .def("sel",
           [](Dataset& self, const py::list& selectors) {
             return DatasetSel(self, selectors);
           },
           py::arg("selectors"))
      .def("__getitem__",
           [](Dataset& self, const std::string& name) {
             auto res = self[name];
             return Unwrap(res);
           },
           py::arg("name"))
      .def("get_variable",
           [](Dataset& self, const std::string& name) {
             auto res = self.get_variable(name);
             return Unwrap(res);
           },
           py::arg("name"))
      .def("get_intervals",
           [](Dataset& self) {
             return IntervalsToPy(Unwrap(self.get_intervals()));
           })
      .def("select_field", &SelectField, py::arg("variable_name"),
           py::arg("field_name"))
      .def("commit_metadata",
           [](Dataset& self) {
             auto fut = self.CommitMetadata();
             Wait(fut);
           })
      .def_property_readonly("metadata",
                             [](const Dataset& self) {
                               return JsonToPy(self.getMetadata());
                             })
      .def_property_readonly(
          "domain", [](const Dataset& self) { return DomainToDict(self.domain); })
      .def_property_readonly(
          "variable_names", [](const Dataset& self) {
            return self.variables.get_iterable_accessor();
          })
      .def_property_readonly(
          "coordinates",
          [](const Dataset& self) { return self.coordinates; })
      .def("__repr__", [](const Dataset& self) {
        return StreamToString(self);
      })
      .def("__str__", [](const Dataset& self) {
        return StreamToString(self);
      })
      .def("__reduce__", [](const py::object& self) {
        const auto& ds = self.cast<const Dataset&>();

        py::list variable_specs;
        for (const auto& name : ds.variables.get_iterable_accessor()) {
          auto var_res = ds.variables.at(name);
          if (!var_res.ok()) {
            ThrowStatus(var_res.status());
          }
          auto spec =
              NormalizeSpecForPickle(Unwrap(var_res.value().get_spec()));
          variable_specs.append(JsonToPy(spec));
        }

        auto metadata = JsonToPy(ds.getMetadata());
        return py::make_tuple(self.attr("__class__"),
                              py::make_tuple(metadata, variable_specs, "open"));
      });

  py::class_<PyWriteFutures>(m, "WriteFutures")
      .def("wait_copy", &PyWriteFutures::wait_copy,
           "Block until copy completes (source no longer needed).")
      .def("wait_commit", &PyWriteFutures::wait_commit,
           "Block until commit/durability completes.")
      .def("wait_all", &PyWriteFutures::wait_all,
           "Block until both copy and commit complete.");

  py::class_<PyVariableData>(m, "VariableDataHandle")
      .def_property(
          "metadata",
          [](const PyVariableData& self) { return JsonToPy(self.data.metadata); },
          [](PyVariableData& self, const py::handle& obj) {
            self.data.metadata = PyToJson(obj);
          })
      .def_property_readonly("domain",
                             [](const PyVariableData& self) {
                               return DomainToDict(self.data.dimensions());
                             })
      .def_property_readonly(
          "dtype",
          [](const PyVariableData& self) {
            return std::string(self.data.dtype().name());
          })
      .def_property_readonly(
          "data",
          [](PyVariableData& self) {
            return SharedArrayToNumpy(self.data.get_data_accessor());
          },
          "NumPy view backed by the VariableData buffer")
      .def("__repr__", [](const PyVariableData& self) {
        return StreamToString(self.data);
      })
      .def("__str__", [](const PyVariableData& self) {
        return StreamToString(self.data);
      })
      .def("to_dict", &PyVariableData::to_dict,
           "Return a dict with metadata, domain, and data.");

  m.def("_unpickle_dataset", &UnpickleDataset);
  m.def("_unpickle_variable", &UnpickleVariable);
}
