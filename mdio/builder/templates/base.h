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
 * `spec()` is this session object. Chunk and unit mutators write the spec in
 * place. `BuildDataset` expands chunk `-1` from the sizes argument and only
 * records `dim_sizes_` after `Build()` succeeds.
 */
class DatasetTemplate {
 public:
  explicit DatasetTemplate(TemplateSpec spec) : spec_(std::move(spec)) {}

  const TemplateSpec& spec() const { return spec_; }

  std::string name() const { return spec_.name; }

  std::string default_variable_name() const {
    return spec_.default_variable_name;
  }

  SeismicDataDomain data_domain() const { return spec_.data_domain; }

  std::vector<std::string> dimension_names() const {
    std::vector<std::string> names;
    names.reserve(spec_.dims.size());
    for (const auto& dim : spec_.dims) {
      names.push_back(dim.name);
    }
    return names;
  }

  std::vector<std::string> spatial_dimension_names() const {
    std::vector<std::string> names = dimension_names();
    if (!names.empty()) {
      names.pop_back();
    }
    return names;
  }

  std::vector<std::string> calculated_dimension_names() const {
    std::vector<std::string> names;
    for (const auto& dim : spec_.dims) {
      if (dim.kind == DimKind::kCalculated) {
        names.push_back(dim.name);
      }
    }
    return names;
  }

  std::vector<std::string> physical_coordinate_names() const {
    return CoordNames(CoordRole::kPhysical);
  }

  std::vector<std::string> logical_coordinate_names() const {
    return CoordNames(CoordRole::kLogical);
  }

  std::vector<std::string> coordinate_names() const {
    std::vector<std::string> names = physical_coordinate_names();
    const std::vector<std::string> logical = logical_coordinate_names();
    names.insert(names.end(), logical.begin(), logical.end());
    return names;
  }

  std::vector<std::string> synthesize_missing_dims() const {
    std::vector<std::string> names;
    for (const auto& dim : spec_.dims) {
      if (dim.kind == DimKind::kSynthesizeIfMissing) {
        names.push_back(dim.name);
      }
    }
    return names;
  }

  const std::vector<int64_t>& dim_sizes() const { return dim_sizes_; }

  const std::vector<int64_t>& stored_chunk_shape() const {
    return spec_.chunks;
  }

  const std::vector<CoordinateSpec>& coordinate_specs() const {
    return spec_.coords;
  }

  std::map<std::string, ScalarType> dim_coordinate_types() const {
    std::map<std::string, ScalarType> types;
    for (const auto& dim : spec_.dims) {
      types.emplace(dim.name, dim.dtype);
    }
    return types;
  }

  nlohmann::json dataset_attributes() const { return spec_.attributes; }

  std::vector<int64_t> full_chunk_shape() const {
    if (dim_sizes_.size() != spec_.dims.size()) {
      return spec_.chunks;
    }
    return ExpandChunks(spec_.chunks, dim_sizes_);
  }

  Result<void> set_full_chunk_shape(const std::vector<int64_t>& shape) {
    if (shape.size() != spec_.dims.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Chunk shape has ", shape.size(),
                       " dimensions, expected ", spec_.dims.size()));
    }
    for (int64_t chunk_size : shape) {
      if (chunk_size != -1 && chunk_size <= 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Chunk size must be positive integer or -1, got ", chunk_size));
      }
    }
    spec_.chunks = shape;
    return absl::OkStatus();
  }

  Result<void> AddUnits(const std::map<std::string, nlohmann::json>& units) {
    for (const auto& [key, unit] : units) {
      if (!IsUnitModel(unit)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unit for '", key, "' is not an instance of an MDIO unit model"));
      }
    }
    for (const auto& [key, unit] : units) {
      spec_.default_units[key] = unit;
    }
    return absl::OkStatus();
  }

  const std::map<std::string, nlohmann::json>& units() const {
    return spec_.default_units;
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
      const std::optional<nlohmann::json>& header_dtype = std::nullopt) {
    if (sizes.size() != spec_.dims.size()) {
      return absl::InvalidArgumentError(absl::StrCat("sizes has ", sizes.size(),
                                                     " entries, expected ",
                                                     spec_.dims.size()));
    }
    if (spec_.chunks.size() != spec_.dims.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Chunk shape has ", spec_.chunks.size(),
                       " dimensions, expected ", spec_.dims.size()));
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

    const std::optional<nlohmann::json> compressor =
        spec_.blosc_on_non_dim_coords
            ? std::optional<nlohmann::json>(DefaultBlosc())
            : std::nullopt;
    for (const auto& coord : spec_.coords) {
      MDIO_RETURN_IF_ERROR(builder.AddCoordinate(
          coord.name, coord.dimensions, coord.dtype, compressor,
          UnitsMetadataOrNull(GetUnitByKey(coord.name))));
    }

    const std::vector<std::string> spatial = spatial_dimension_names();
    const std::vector<std::string> coords = coordinate_names();
    const std::vector<std::string> dims = dimension_names();

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
      MDIO_RETURN_IF_ERROR(builder.AddVariable("headers", spatial,
                                               *header_dtype, DefaultBlosc(),
                                               coords, header_metadata));
    }
    MDIO_ASSIGN_OR_RETURN(nlohmann::json dataset, builder.Build());
    dim_sizes_ = sizes;
    return dataset;
  }

 private:
  std::vector<std::string> CoordNames(CoordRole role) const {
    std::vector<std::string> names;
    for (const auto& coord : spec_.coords) {
      if (coord.role == role) {
        names.push_back(coord.name);
      }
    }
    return names;
  }

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
