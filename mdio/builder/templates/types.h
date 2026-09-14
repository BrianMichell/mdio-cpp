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

#ifndef MDIO_BUILDER_TEMPLATES_TYPES_H_
#define MDIO_BUILDER_TEMPLATES_TYPES_H_

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "mdio/builder/schemas.h"
#include "mdio/impl.h"

// clang-format off
#include <nlohmann/json.hpp>  // NOLINT
// clang-format on

namespace mdio {
namespace builder {

enum class SeismicDataDomain { kTime, kDepth };

enum class CdpGatherDomain { kOffset, kAngle };

enum class CoordRole { kPhysical, kLogical };

/**
 * @brief How a dimension is materialized as a 1-d coordinate.
 *
 * `kStored`: emit a 1-d coordinate (default).
 * `kCalculated`: omit the coordinate (e.g. `shot_index`).
 * `kSynthesizeIfMissing`: emit a coordinate; ingestion may invent the dim
 * (e.g. `component`).
 */
enum class DimKind {
  kStored,
  kCalculated,
  kSynthesizeIfMissing,
};

/**
 * @brief One dataset dimension.
 */
struct DimSpec {
  std::string name;
  ScalarType dtype = ScalarType::kInt32;
  DimKind kind = DimKind::kStored;
};

/**
 * @brief Non-dimension coordinate. Role drives physical vs logical name lists.
 */
struct CoordinateSpec {
  std::string name;
  std::vector<std::string> dimensions;
  ScalarType dtype = ScalarType::kFloat64;
  CoordRole role = CoordRole::kPhysical;
};

/**
 * @brief Complete description of one canonical (or custom) dataset template.
 *
 * Single source of truth for catalog queries. Last dimension is the sample
 * (time/depth) axis; `spatial_dimension_names()` is every dim except that one.
 */
struct TemplateSpec {
  std::string name;
  SeismicDataDomain data_domain = SeismicDataDomain::kTime;
  std::vector<DimSpec> dims;
  std::vector<CoordinateSpec> coords;
  std::vector<int64_t> chunks;
  nlohmann::json attributes = nlohmann::json::object();
  std::string default_variable_name = "amplitude";
  std::map<std::string, nlohmann::json> default_units;
  std::optional<nlohmann::json> non_dim_coord_compressor = DefaultBlosc();

  std::vector<std::string> dimension_names() const {
    return DimNames(/*only=*/std::nullopt);
  }

  // Last dim is the sample axis. Empty if the spec has no dimensions.
  std::vector<std::string> spatial_dimension_names() const {
    std::vector<std::string> names = dimension_names();
    if (!names.empty()) {
      names.pop_back();
    }
    return names;
  }

  std::vector<std::string> calculated_dimension_names() const {
    return DimNames(DimKind::kCalculated);
  }

  std::vector<std::string> synthesize_missing_dims() const {
    return DimNames(DimKind::kSynthesizeIfMissing);
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

  std::map<std::string, ScalarType> dim_coordinate_types() const {
    std::map<std::string, ScalarType> types;
    for (const auto& dim : dims) {
      types.emplace(dim.name, dim.dtype);
    }
    return types;
  }

  const DimSpec* FindDim(std::string_view name) const {
    for (const auto& dim : dims) {
      if (dim.name == name) {
        return &dim;
      }
    }
    return nullptr;
  }

 private:
  std::vector<std::string> DimNames(std::optional<DimKind> only) const {
    std::vector<std::string> names;
    for (const auto& dim : dims) {
      if (only.has_value() && dim.kind != *only) {
        continue;
      }
      names.push_back(dim.name);
    }
    return names;
  }

  std::vector<std::string> CoordNames(CoordRole role) const {
    std::vector<std::string> names;
    for (const auto& coord : coords) {
      if (coord.role == role) {
        names.push_back(coord.name);
      }
    }
    return names;
  }
};

inline const char* ToString(SeismicDataDomain domain) {
  return domain == SeismicDataDomain::kTime ? "time" : "depth";
}

inline const char* ToString(CdpGatherDomain domain) {
  return domain == CdpGatherDomain::kOffset ? "offset" : "angle";
}

inline Result<void> ValidateChunkShape(const std::vector<int64_t>& chunks,
                                       size_t expected_rank) {
  if (chunks.size() != expected_rank) {
    return absl::InvalidArgumentError(
        absl::StrCat("Chunk shape has ", chunks.size(),
                     " dimensions, expected ", expected_rank));
  }
  for (int64_t chunk_size : chunks) {
    if (chunk_size != -1 && chunk_size <= 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Chunk size must be positive integer or -1, got ", chunk_size));
    }
  }
  return absl::OkStatus();
}

inline Result<void> ValidateTemplateSpec(const TemplateSpec& spec) {
  if (spec.name.empty()) {
    return absl::InvalidArgumentError("Template name must be non-empty");
  }
  if (spec.dims.empty()) {
    return absl::InvalidArgumentError(
        "Template must have at least one dimension");
  }
  auto chunks_ok = ValidateChunkShape(spec.chunks, spec.dims.size());
  if (!chunks_ok.ok()) {
    return chunks_ok;
  }
  for (size_t i = 0; i < spec.dims.size(); ++i) {
    if (spec.dims[i].name.empty()) {
      return absl::InvalidArgumentError("Dimension name must be non-empty");
    }
    for (size_t j = i + 1; j < spec.dims.size(); ++j) {
      if (spec.dims[i].name == spec.dims[j].name) {
        return absl::InvalidArgumentError(
            absl::StrCat("Duplicate dimension name '", spec.dims[i].name, "'"));
      }
    }
  }
  for (size_t i = 0; i < spec.coords.size(); ++i) {
    const auto& coord = spec.coords[i];
    if (coord.name.empty()) {
      return absl::InvalidArgumentError("Coordinate name must be non-empty");
    }
    if (spec.FindDim(coord.name) != nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Coordinate '", coord.name, "' collides with a dimension name"));
    }
    for (size_t j = i + 1; j < spec.coords.size(); ++j) {
      if (coord.name == spec.coords[j].name) {
        return absl::InvalidArgumentError(
            absl::StrCat("Duplicate coordinate name '", coord.name, "'"));
      }
    }
    if (coord.dimensions.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Coordinate '", coord.name, "' has no dimensions"));
    }
    for (const auto& dim_name : coord.dimensions) {
      if (spec.FindDim(dim_name) == nullptr) {
        return absl::InvalidArgumentError(
            absl::StrCat("Coordinate '", coord.name, "' references unknown ",
                         "dimension '", dim_name, "'"));
      }
    }
  }
  auto units_ok = ValidateUnits(spec.default_units);
  if (!units_ok.ok()) {
    return units_ok;
  }
  return absl::OkStatus();
}

inline Result<SeismicDataDomain> ParseSeismicDataDomain(
    std::string_view domain) {
  const std::string lower = absl::AsciiStrToLower(domain);
  if (lower == "time") {
    return SeismicDataDomain::kTime;
  }
  if (lower == "depth") {
    return SeismicDataDomain::kDepth;
  }
  return absl::InvalidArgumentError("domain must be 'depth' or 'time'");
}

inline Result<CdpGatherDomain> ParseCdpGatherDomain(std::string_view domain) {
  const std::string lower = absl::AsciiStrToLower(domain);
  if (lower == "offset") {
    return CdpGatherDomain::kOffset;
  }
  if (lower == "angle") {
    return CdpGatherDomain::kAngle;
  }
  return absl::InvalidArgumentError("gather_type must be 'offset' or 'angle'");
}

}  // namespace builder
}  // namespace mdio

#endif  // MDIO_BUILDER_TEMPLATES_TYPES_H_
