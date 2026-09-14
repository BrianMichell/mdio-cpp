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

#ifndef MDIO_BUILDER_TEMPLATES_BASE_H_
#define MDIO_BUILDER_TEMPLATES_BASE_H_

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mdio/builder/dataset_builder.h"
#include "mdio/builder/templates/types.h"
#include "mdio/impl.h"

// clang-format off
#include <nlohmann/json.hpp>  // NOLINT
// clang-format on

namespace mdio {
namespace builder {

/**
 * @brief Mutable working copy of a `TemplateSpec`.
 *
 * Catalog queries live on `spec()`. This object owns chunk/unit mutation and
 * `BuildDataset`. `full_chunk_shape()` expands `-1` only after a successful
 * build (sizes remembered in `dim_sizes_`).
 */
class DatasetTemplate {
 public:
  explicit DatasetTemplate(TemplateSpec spec) : spec_(std::move(spec)) {}

  const TemplateSpec& spec() const { return spec_; }

  std::string name() const { return spec_.name; }

  const std::vector<int64_t>& stored_chunk_shape() const {
    return spec_.chunks;
  }

  std::vector<int64_t> full_chunk_shape() const {
    if (dim_sizes_.size() != spec_.dims.size()) {
      return spec_.chunks;
    }
    return ExpandChunks(spec_.chunks, dim_sizes_);
  }

  Result<void> set_full_chunk_shape(const std::vector<int64_t>& shape) {
    auto valid = ValidateChunkShape(shape, spec_.dims.size());
    if (!valid.ok()) {
      return valid;
    }
    spec_.chunks = shape;
    return absl::OkStatus();
  }

  Result<void> AddUnits(const std::map<std::string, nlohmann::json>& units) {
    auto valid = ValidateUnits(units);
    if (!valid.ok()) {
      return valid;
    }
    for (const auto& [key, unit] : units) {
      spec_.default_units[key] = unit;
    }
    return absl::OkStatus();
  }

  std::optional<nlohmann::json> GetUnitByKey(const std::string& key) const {
    const auto it = spec_.default_units.find(key);
    if (it == spec_.default_units.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  Result<nlohmann::json> BuildDataset(
      const std::string& name, const std::vector<int64_t>& sizes,
      const std::optional<StructuredType>& header_dtype = std::nullopt) {
    MDIO_RETURN_IF_ERROR(ValidateTemplateSpec(spec_));
    if (sizes.size() != spec_.dims.size()) {
      return absl::InvalidArgumentError(absl::StrCat("sizes has ", sizes.size(),
                                                     " entries, expected ",
                                                     spec_.dims.size()));
    }
    const std::vector<int64_t> chunks = ExpandChunks(spec_.chunks, sizes);

    nlohmann::json attributes = spec_.attributes;
    if (attributes.is_null()) {
      attributes = nlohmann::json::object();
    }
    attributes["defaultVariableName"] = spec_.default_variable_name;

    MDIODatasetBuilder builder(name, attributes);
    for (size_t i = 0; i < spec_.dims.size(); ++i) {
      MDIO_RETURN_IF_ERROR(builder.AddDimension(spec_.dims[i].name, sizes[i]));
    }

    for (const auto& dim : spec_.dims) {
      if (dim.kind == DimKind::kCalculated) {
        continue;
      }
      MDIO_RETURN_IF_ERROR(builder.AddCoordinate(
          dim.name, {dim.name}, dim.dtype, /*compressor=*/std::nullopt,
          UnitsMetadataOrNull(GetUnitByKey(dim.name))));
    }

    for (const auto& coord : spec_.coords) {
      MDIO_RETURN_IF_ERROR(
          builder.AddCoordinate(coord.name, coord.dimensions, coord.dtype,
                                spec_.non_dim_coord_compressor,
                                UnitsMetadataOrNull(GetUnitByKey(coord.name))));
    }

    const std::vector<std::string> spatial = spec_.spatial_dimension_names();
    const std::vector<std::string> coords = spec_.coordinate_names();
    const std::vector<std::string> dims = spec_.dimension_names();

    nlohmann::json amplitude_metadata = {
        {"chunkGrid", RegularChunkGridJson(chunks)}};
    if (auto unit = GetUnitByKey(spec_.default_variable_name);
        unit.has_value()) {
      amplitude_metadata["unitsV1"] = *unit;
    }
    MDIO_RETURN_IF_ERROR(builder.AddVariable(
        spec_.default_variable_name, dims, DataTypeJson(ScalarType::kFloat32),
        DefaultBlosc(), coords, amplitude_metadata));
    MDIO_RETURN_IF_ERROR(builder.AddVariable("trace_mask", spatial,
                                             DataTypeJson(ScalarType::kBool),
                                             DefaultBlosc(), coords));

    if (header_dtype.has_value()) {
      std::vector<int64_t> spatial_chunks(
          chunks.begin(), chunks.empty() ? chunks.end() : chunks.end() - 1);
      nlohmann::json header_metadata = {
          {"chunkGrid", RegularChunkGridJson(spatial_chunks)}};
      MDIO_RETURN_IF_ERROR(
          builder.AddVariable("headers", spatial, header_dtype->ToJson(),
                              DefaultBlosc(), coords, header_metadata));
    }
    MDIO_ASSIGN_OR_RETURN(nlohmann::json dataset, builder.Build());
    dim_sizes_ = sizes;
    return dataset;
  }

 private:
  static std::vector<int64_t> ExpandChunks(const std::vector<int64_t>& chunks,
                                           const std::vector<int64_t>& sizes) {
    std::vector<int64_t> expanded;
    expanded.reserve(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
      expanded.push_back(chunks[i] == -1 ? sizes[i] : chunks[i]);
    }
    return expanded;
  }

  TemplateSpec spec_;
  std::vector<int64_t> dim_sizes_;
};

}  // namespace builder
}  // namespace mdio

#endif  // MDIO_BUILDER_TEMPLATES_BASE_H_
