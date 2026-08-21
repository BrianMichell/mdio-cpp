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

#ifndef MDIO_BUILDER_DATASET_BUILDER_H_
#define MDIO_BUILDER_DATASET_BUILDER_H_

#include <ctime>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mdio/builder/schemas.h"
#include "mdio/impl.h"

// clang-format off
#include <nlohmann/json.hpp>  // NOLINT
// clang-format on

namespace mdio {
namespace builder {

struct NamedDimension {
  std::string name;
  int64_t size;
};

struct Variable {
  std::string name;
  std::optional<std::string> long_name;
  std::vector<NamedDimension> dimensions;
  nlohmann::json data_type;
  std::optional<nlohmann::json> compressor;
  std::optional<std::vector<std::string>> coordinates;
  std::optional<nlohmann::json> metadata;
};

inline std::string UtcNowIso8601() {
  std::time_t now = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

/**
 * @brief Assembles MDIO v1 Dataset JSON.
 *
 * Coordinates are emitted as string names so `Dataset::from_json` can consume
 * the result.
 */
class MDIODatasetBuilder {
 public:
  explicit MDIODatasetBuilder(std::string name,
                              nlohmann::json attributes = nlohmann::json())
      : name_(std::move(name)),
        attributes_(std::move(attributes)),
        api_version_(kBuilderApiVersion),
        created_on_(UtcNowIso8601()) {}

  Result<void> AddDimension(const std::string& name, int64_t size) {
    if (name.empty()) {
      return absl::InvalidArgumentError("'name' must be a non-empty string");
    }
    if (size <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Dimension size must be positive, got ", size));
    }
    if (FindDimension(name) != nullptr) {
      return absl::InvalidArgumentError(
          "Adding dimension with the same name twice is not allowed");
    }
    dimensions_.push_back({name, size});
    return absl::OkStatus();
  }

  Result<void> AddCoordinate(
      const std::string& name, const std::vector<std::string>& dimensions,
      ScalarType data_type,
      const std::optional<nlohmann::json>& compressor = std::nullopt,
      const std::optional<nlohmann::json>& metadata = std::nullopt,
      const std::optional<std::string>& long_name = std::nullopt) {
    if (dimensions_.empty()) {
      return absl::InvalidArgumentError(
          "Must add at least one dimension before adding coordinates");
    }
    if (name.empty()) {
      return absl::InvalidArgumentError("'name' must be a non-empty string");
    }
    if (dimensions.empty()) {
      return absl::InvalidArgumentError(
          "'dimensions' must be a non-empty list");
    }
    if (HasCoordinate(name)) {
      return absl::InvalidArgumentError(
          "Adding coordinate with the same name twice is not allowed");
    }

    std::vector<std::string> resolved;
    resolved.reserve(dimensions.size());
    for (const auto& dim_name : dimensions) {
      if (FindDimension(dim_name) == nullptr) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Pre-existing dimension named '", dim_name, "' is not found"));
      }
      resolved.push_back(dim_name);
    }

    // Variable first so a failed AddVariable leaves no orphan Coordinate.
    // The variable may list itself as a coordinate (it is the coordinate).
    auto added =
        AddVariable(name, resolved, DataTypeJson(data_type), compressor,
                    std::vector<std::string>{name}, metadata, long_name);
    if (!added.ok()) {
      return added;
    }
    coordinate_names_.push_back(name);
    return absl::OkStatus();
  }

  Result<void> AddVariable(
      const std::string& name, const std::vector<std::string>& dimensions,
      const nlohmann::json& data_type,
      const std::optional<nlohmann::json>& compressor = std::nullopt,
      const std::optional<std::vector<std::string>>& coordinates = std::nullopt,
      const std::optional<nlohmann::json>& metadata = std::nullopt,
      const std::optional<std::string>& long_name = std::nullopt) {
    if (dimensions_.empty()) {
      return absl::InvalidArgumentError(
          "Must add at least one dimension before adding variables");
    }
    if (name.empty()) {
      return absl::InvalidArgumentError("'name' must be a non-empty string");
    }
    if (dimensions.empty()) {
      return absl::InvalidArgumentError(
          "'dimensions' must be a non-empty list");
    }
    if (FindVariable(name) != nullptr) {
      return absl::InvalidArgumentError(
          "Adding variable with the same name twice is not allowed");
    }

    std::vector<NamedDimension> named;
    named.reserve(dimensions.size());
    for (const auto& dim_name : dimensions) {
      const NamedDimension* dim = FindDimension(dim_name);
      if (dim == nullptr) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Pre-existing dimension named '", dim_name, "' is not found"));
      }
      named.push_back(*dim);
    }

    if (coordinates.has_value()) {
      for (const auto& coord_name : *coordinates) {
        if (coord_name == name) {
          continue;
        }
        if (!HasCoordinate(coord_name)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Pre-existing coordinate named '", coord_name, "' is not found"));
        }
      }
    }

    variables_.push_back({name, long_name, std::move(named), data_type,
                          compressor, coordinates, metadata});
    return absl::OkStatus();
  }

  Result<nlohmann::json> Build() const {
    if (dimensions_.empty()) {
      return absl::InvalidArgumentError(
          "Must add at least one dimension before building");
    }

    nlohmann::json metadata = {{"name", name_},
                               {"apiVersion", api_version_},
                               {"createdOn", created_on_}};
    if (!attributes_.is_null() &&
        !(attributes_.is_object() && attributes_.empty())) {
      metadata["attributes"] = attributes_;
    }

    nlohmann::json variables = nlohmann::json::array();
    for (const auto& var : variables_) {
      variables.push_back(VariableToJson(var));
    }
    return nlohmann::json{{"metadata", std::move(metadata)},
                          {"variables", std::move(variables)}};
  }

  const std::vector<NamedDimension>& dimensions() const { return dimensions_; }
  const std::vector<std::string>& coordinate_names() const {
    return coordinate_names_;
  }
  const std::vector<Variable>& variables() const { return variables_; }
  const std::string& name() const { return name_; }

  const NamedDimension* FindDimension(std::string_view name) const {
    for (const auto& dim : dimensions_) {
      if (dim.name == name) {
        return &dim;
      }
    }
    return nullptr;
  }

  bool HasCoordinate(std::string_view name) const {
    for (const auto& coord : coordinate_names_) {
      if (coord == name) {
        return true;
      }
    }
    return false;
  }

  const Variable* FindVariable(std::string_view name) const {
    for (const auto& var : variables_) {
      if (var.name == name) {
        return &var;
      }
    }
    return nullptr;
  }

 private:
  static nlohmann::json VariableToJson(const Variable& var) {
    nlohmann::json json;
    json["name"] = var.name;
    json["dataType"] = var.data_type;
    json["dimensions"] = nlohmann::json::array();
    for (const auto& dim : var.dimensions) {
      json["dimensions"].push_back({{"name", dim.name}, {"size", dim.size}});
    }
    if (var.long_name.has_value()) {
      json["longName"] = *var.long_name;
    }
    if (var.compressor.has_value()) {
      json["compressor"] = *var.compressor;
    }
    if (var.coordinates.has_value()) {
      json["coordinates"] = *var.coordinates;
    }
    if (var.metadata.has_value() && !var.metadata->is_null() &&
        !var.metadata->empty()) {
      json["metadata"] = *var.metadata;
    }
    return json;
  }

  std::string name_;
  nlohmann::json attributes_;
  std::string api_version_;
  std::string created_on_;
  std::vector<NamedDimension> dimensions_;
  std::vector<std::string> coordinate_names_;
  std::vector<Variable> variables_;
};

}  // namespace builder
}  // namespace mdio

#endif  // MDIO_BUILDER_DATASET_BUILDER_H_
