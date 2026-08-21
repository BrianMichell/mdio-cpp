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
#include <string>
#include <string_view>
#include <vector>

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
 * Single source of truth. `DatasetTemplate` emits coordinates from `dims` +
 * `coords`. Physical / logical names are derived from `coords` role.
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
  bool blosc_on_non_dim_coords = true;
};

inline const char* ToString(SeismicDataDomain domain) {
  return domain == SeismicDataDomain::kTime ? "time" : "depth";
}

inline const char* ToString(CdpGatherDomain domain) {
  return domain == CdpGatherDomain::kOffset ? "offset" : "angle";
}

inline Result<SeismicDataDomain> ParseSeismicDataDomain(
    std::string_view domain) {
  const std::string lower = AsciiLower(domain);
  if (lower == "time") {
    return SeismicDataDomain::kTime;
  }
  if (lower == "depth") {
    return SeismicDataDomain::kDepth;
  }
  return absl::InvalidArgumentError("domain must be 'depth' or 'time'");
}

inline Result<CdpGatherDomain> ParseCdpGatherDomain(std::string_view domain) {
  const std::string lower = AsciiLower(domain);
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
