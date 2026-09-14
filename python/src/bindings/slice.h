// Copyright 2026 TGS

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PYTHON_SRC_BINDINGS_SLICE_H_
#define PYTHON_SRC_BINDINGS_SLICE_H_

#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "bindings/convert.h"
#include "mdio/dataset.h"
#include "mdio/utils/trim.h"

namespace mdio_py {

inline void RequireUnitStep(mdio::Index step) {
  if (step != 1) {
    throw MdioError("MDIO only supports step=1 slices");
  }
}

inline bool TryAsIndex(const py::handle& obj, mdio::Index* out) {
  if (!PyIndex_Check(obj.ptr())) {
    return false;
  }
  *out = obj.cast<mdio::Index>();
  return true;
}

// DimensionIdentifier stores a string_view. Python-facing ranges own the
// label string so the view stays valid for the object's lifetime.
struct PyRange {
  std::string label;
  mdio::Index start = 0;
  mdio::Index stop = 0;
  mdio::Index step = 1;

  PyRange() = default;
  PyRange(std::string label_in, mdio::Index start_in, mdio::Index stop_in,
          mdio::Index step_in = 1)
      : label(std::move(label_in)),
        start(start_in),
        stop(stop_in),
        step(step_in) {
    RequireUnitStep(step);
  }

  mdio::RangeDescriptor<mdio::Index> ToDescriptor() const {
    return {label, start, stop, step};
  }
};

// Coordinate value for sel. Holds a Python object so float coords work.
struct PyValue {
  std::string label;
  py::object value;

  PyValue() = default;
  PyValue(std::string label_in, py::object value_in)
      : label(std::move(label_in)), value(std::move(value_in)) {}
};

struct SelRange {
  std::string label;
  py::object start;
  py::object stop;
};

using SelOp = std::variant<SelRange, PyValue>;

inline bool LookupDimension(const IndexDomainInfo& domain,
                            const std::string& label, mdio::Index* origin,
                            mdio::Index* size) {
  for (mdio::DimensionIndex i = 0; i < domain.rank; ++i) {
    if (domain.labels[i] == label) {
      *origin = domain.origin[i];
      *size = domain.shape[i];
      return true;
    }
  }
  return false;
}

inline std::string ResolveLabel(
    const std::optional<std::string>& override_label,
    const std::string& from_obj) {
  if (override_label.has_value()) {
    return *override_label;
  }
  if (from_obj.empty()) {
    throw MdioError("Slice missing label");
  }
  return from_obj;
}

inline PyRange SliceToRange(const std::string& label, const py::slice& slice,
                            const IndexDomainInfo& domain) {
  py::object start_obj = slice.attr("start");
  py::object stop_obj = slice.attr("stop");
  py::object step_obj = slice.attr("step");
  mdio::Index origin = 0;
  mdio::Index size = 0;
  const bool found = LookupDimension(domain, label, &origin, &size);
  if ((start_obj.is_none() || stop_obj.is_none()) && !found) {
    throw MdioError("Cannot infer slice bounds for dimension '" + label + "'");
  }
  const mdio::Index start =
      start_obj.is_none() ? origin : start_obj.cast<mdio::Index>();
  const mdio::Index stop =
      stop_obj.is_none() ? origin + size : stop_obj.cast<mdio::Index>();
  const mdio::Index step =
      step_obj.is_none() ? 1 : step_obj.cast<mdio::Index>();
  return PyRange{label, start, stop, step};
}

// isel only. kwargs label wins over any label on the object.
inline PyRange ParseSliceInput(const py::handle& obj,
                               const std::optional<std::string>& label,
                               const IndexDomainInfo& domain) {
  if (py::isinstance<PyRange>(obj)) {
    auto desc = obj.cast<PyRange>();
    desc.label = ResolveLabel(label, desc.label);
    return desc;
  }
  if (py::isinstance<py::slice>(obj)) {
    if (!label.has_value()) {
      throw MdioError("Bare slice needs a dimension name");
    }
    return SliceToRange(*label, obj.cast<py::slice>(), domain);
  }
  mdio::Index index = 0;
  if (TryAsIndex(obj, &index)) {
    if (!label.has_value()) {
      throw MdioError("Bare index needs a dimension name");
    }
    return PyRange{*label, index, index + 1, 1};
  }
  if (py::isinstance<py::tuple>(obj)) {
    if (!label.has_value()) {
      throw MdioError("Bare (start, stop) needs a dimension name");
    }
    auto seq = obj.cast<py::sequence>();
    if (seq.size() < 2 || seq.size() > 3) {
      throw MdioError("Slice sequence must be (start, stop[, step])");
    }
    const mdio::Index start = seq[0].cast<mdio::Index>();
    const mdio::Index stop = seq[1].cast<mdio::Index>();
    const mdio::Index step = seq.size() == 3 ? seq[2].cast<mdio::Index>() : 1;
    return PyRange{*label, start, stop, step};
  }
  throw MdioError("Cannot convert argument to a slice descriptor");
}

inline std::vector<PyRange> ParseISelArgs(const py::args& args,
                                          const py::kwargs& kwargs,
                                          const IndexDomainInfo& domain) {
  std::vector<PyRange> slices;
  for (const auto& arg : args) {
    if (py::isinstance<py::list>(arg)) {
      for (const auto& item : arg) {
        slices.push_back(ParseSliceInput(item, std::nullopt, domain));
      }
    } else {
      slices.push_back(ParseSliceInput(arg, std::nullopt, domain));
    }
  }
  for (const auto& item : kwargs) {
    slices.push_back(
        ParseSliceInput(item.second, item.first.cast<std::string>(), domain));
  }
  return slices;
}

inline std::vector<mdio::RangeDescriptor<mdio::Index>> ToDescriptors(
    const std::vector<PyRange>& ranges) {
  std::vector<mdio::RangeDescriptor<mdio::Index>> slices;
  slices.reserve(ranges.size());
  for (const auto& range : ranges) {
    slices.push_back(range.ToDescriptor());
  }
  return slices;
}

inline mdio::Dataset ApplyISel(mdio::Dataset& dataset,
                               const std::vector<PyRange>& slices) {
  if (slices.empty()) {
    return dataset;
  }
  return CheckResult(dataset.isel(ToDescriptors(slices)));
}

inline SelOp ParseSelInput(const std::string& label, const py::handle& value) {
  if (py::isinstance<py::list>(value)) {
    throw MdioError("ListDescriptor sel is not implemented");
  }
  if (py::isinstance<py::slice>(value)) {
    auto slice = value.cast<py::slice>();
    py::object start = slice.attr("start");
    py::object stop = slice.attr("stop");
    py::object step = slice.attr("step");
    if (!step.is_none()) {
      RequireUnitStep(step.cast<mdio::Index>());
    }
    if (start.is_none() || stop.is_none()) {
      throw MdioError("sel range requires concrete start and stop values");
    }
    return SelRange{label, start, stop};
  }
  if (py::isinstance<PyRange>(value)) {
    auto desc = value.cast<PyRange>();
    return SelRange{label, py::int_(desc.start), py::int_(desc.stop)};
  }
  if (py::isinstance<py::tuple>(value)) {
    auto seq = value.cast<py::sequence>();
    if (seq.size() < 2 || seq.size() > 3) {
      throw MdioError("sel range tuple must be (start, stop[, step])");
    }
    if (seq.size() == 3) {
      RequireUnitStep(seq[2].cast<mdio::Index>());
    }
    return SelRange{label, seq[0], seq[1]};
  }
  if (py::isinstance<PyValue>(value)) {
    return PyValue{label, value.cast<PyValue>().value};
  }
  return PyValue{label, py::reinterpret_borrow<py::object>(value)};
}

inline std::vector<SelOp> ParseSelArgs(const py::args& args,
                                       const py::kwargs& kwargs) {
  std::vector<SelOp> ops;
  for (const auto& arg : args) {
    if (py::isinstance<py::dict>(arg)) {
      for (auto item : arg.cast<py::dict>()) {
        ops.push_back(
            ParseSelInput(item.first.cast<std::string>(), item.second));
      }
    } else if (py::isinstance<PyValue>(arg)) {
      auto desc = arg.cast<PyValue>();
      ops.push_back(ParseSelInput(desc.label, desc.value));
    } else if (py::isinstance<PyRange>(arg)) {
      auto desc = arg.cast<PyRange>();
      ops.push_back(ParseSelInput(desc.label, arg));
    } else {
      throw MdioError(
          "sel positional arguments must be dicts, Value, or Range");
    }
  }
  for (const auto& item : kwargs) {
    ops.push_back(ParseSelInput(item.first.cast<std::string>(), item.second));
  }
  return ops;
}

inline mdio::Dataset ApplyOneSel(mdio::Dataset dataset, const SelOp& op) {
  return std::visit(
      [&](const auto& spec) {
        using Spec = std::decay_t<decltype(spec)>;
        auto variable = CheckResult(dataset.variables.at(spec.label));
        const mdio::DataType dtype = variable.dtype();
        if constexpr (std::is_same_v<Spec, SelRange>) {
          return VisitNumericDtype(dtype, [&](auto tag) {
            using T = decltype(tag);
            mdio::RangeDescriptor<T> desc{spec.label,
                                          CastNumeric<T>(spec.start),
                                          CastNumeric<T>(spec.stop), 1};
            return Await([&] { return dataset.sel(desc); });
          });
        } else {
          return VisitNumericDtype(dtype, [&](auto tag) {
            using T = decltype(tag);
            mdio::ValueDescriptor<T> desc{spec.label,
                                          CastNumeric<T>(spec.value)};
            return Await([&] { return dataset.sel(desc); });
          });
        }
      },
      op);
}

inline mdio::Dataset ApplySel(mdio::Dataset dataset,
                              const std::vector<SelOp>& ops) {
  for (const auto& op : ops) {
    dataset = ApplyOneSel(std::move(dataset), op);
  }
  return dataset;
}

struct TrimStop {
  std::string label;
  mdio::Index stop;
};

// Trim only applies on-disk stop. Accept {label: integer} only.
inline std::vector<TrimStop> ParseTrimSlices(const py::object& slices) {
  if (!py::isinstance<py::dict>(slices)) {
    throw MdioError("trim slices must be {label: stop}");
  }
  std::vector<TrimStop> parsed;
  for (auto item : slices.cast<py::dict>()) {
    const std::string label = item.first.cast<std::string>();
    mdio::Index stop = 0;
    if (!TryAsIndex(item.second, &stop)) {
      throw MdioError("trim stop for '" + label + "' must be an integer");
    }
    parsed.push_back({label, stop});
  }
  return parsed;
}

template <std::size_t... I>
mdio::Future<void> TrimPack(
    const std::string& path, bool delete_sliced_out_chunks,
    const std::vector<mdio::RangeDescriptor<mdio::Index>>& slices,
    std::index_sequence<I...>) {
  return mdio::utils::TrimDataset(path, delete_sliced_out_chunks, slices[I]...);
}

inline void TrimWithVector(const std::string& path,
                           bool delete_sliced_out_chunks,
                           const std::vector<TrimStop>& stops) {
  if (stops.empty()) {
    return;
  }
  if (stops.size() > mdio::internal::kMaxNumSlices) {
    throw MdioError("Too many trim slices; maximum is " +
                    std::to_string(mdio::internal::kMaxNumSlices));
  }
  std::vector<mdio::RangeDescriptor<mdio::Index>> slices;
  slices.reserve(mdio::internal::kMaxNumSlices);
  for (const auto& stop : stops) {
    slices.push_back({stop.label, 0, stop.stop, 1});
  }
  while (slices.size() < mdio::internal::kMaxNumSlices) {
    slices.push_back({mdio::internal::kInertSliceKey, 0, 1, 1});
  }
  Await([&] {
    return TrimPack(path, delete_sliced_out_chunks, slices,
                    std::make_index_sequence<mdio::internal::kMaxNumSlices>{});
  });
}

}  // namespace mdio_py

#endif  // PYTHON_SRC_BINDINGS_SLICE_H_
