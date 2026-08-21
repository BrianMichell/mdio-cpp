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

#ifndef MDIO_BUILDER_SCHEMAS_H_
#define MDIO_BUILDER_SCHEMAS_H_

#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "mdio/impl.h"

// clang-format off
#include <nlohmann/json.hpp>  // NOLINT
// clang-format on

namespace mdio {
namespace builder {

// Matches `kSchemaVersion` in dataset_schema.h without pulling that file in.
inline constexpr std::string_view kBuilderApiVersion = "1.2.0";

/**
 * @brief Scalar array data types from the MDIO v1 schema.
 * Values match mdio-python `ScalarType`.
 */
enum class ScalarType {
  kBool,
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kUint8,
  kUint16,
  kUint32,
  kUint64,
  kFloat16,
  kFloat32,
  kFloat64,
  kFloat128,
  kComplex64,
  kComplex128,
  kComplex256,
  kBytes240,
};

struct ScalarTypeEntry {
  ScalarType type;
  const char* name;
};

inline constexpr ScalarTypeEntry kScalarTypeEntries[] = {
    {ScalarType::kBool, "bool"},
    {ScalarType::kInt8, "int8"},
    {ScalarType::kInt16, "int16"},
    {ScalarType::kInt32, "int32"},
    {ScalarType::kInt64, "int64"},
    {ScalarType::kUint8, "uint8"},
    {ScalarType::kUint16, "uint16"},
    {ScalarType::kUint32, "uint32"},
    {ScalarType::kUint64, "uint64"},
    {ScalarType::kFloat16, "float16"},
    {ScalarType::kFloat32, "float32"},
    {ScalarType::kFloat64, "float64"},
    {ScalarType::kFloat128, "float128"},
    {ScalarType::kComplex64, "complex64"},
    {ScalarType::kComplex128, "complex128"},
    {ScalarType::kComplex256, "complex256"},
    {ScalarType::kBytes240, "V240"},
};

/**
 * @brief Returns the MDIO schema string for a scalar type.
 */
inline const char* ToString(ScalarType type) {
  for (const auto& entry : kScalarTypeEntries) {
    if (entry.type == type) {
      return entry.name;
    }
  }
  ABSL_UNREACHABLE();
}

/**
 * @brief Parses an MDIO schema dtype string into a ScalarType.
 */
inline Result<ScalarType> ParseScalarType(std::string_view value) {
  for (const auto& entry : kScalarTypeEntries) {
    if (value == entry.name) {
      return entry.type;
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown scalar type: ", value));
}

/**
 * @brief One field of a structured (record) dtype.
 */
struct StructuredField {
  std::string name;
  ScalarType format;
};

/**
 * @brief Structured array type with packed fields.
 */
struct StructuredType {
  std::vector<StructuredField> fields;

  nlohmann::json ToJson() const {
    nlohmann::json out;
    out["fields"] = nlohmann::json::array();
    for (const auto& field : fields) {
      out["fields"].push_back(
          {{"name", field.name}, {"format", ToString(field.format)}});
    }
    return out;
  }
};

/**
 * @brief JSON dtype payload: a scalar string or a structured `fields` object.
 */
inline nlohmann::json DataTypeJson(ScalarType type) { return ToString(type); }

inline nlohmann::json DataTypeJson(const StructuredType& type) {
  return type.ToJson();
}

/**
 * @brief Default Blosc compressor used by canonical templates (zstd).
 */
inline nlohmann::json DefaultBlosc() {
  return {{"name", "blosc"}, {"cname", "zstd"}};
}

/**
 * @brief Regular chunk grid metadata block (`chunkGrid` value).
 */
inline nlohmann::json RegularChunkGridJson(
    const std::vector<int64_t>& chunk_shape) {
  return {{"name", "regular"},
          {"configuration", {{"chunkShape", chunk_shape}}}};
}

/**
 * @brief Unit model helpers matching mdio-python AllUnitModel JSON.
 */
inline nlohmann::json LengthUnit(std::string_view unit) {
  return {{"length", std::string(unit)}};
}

inline nlohmann::json TimeUnit(std::string_view unit) {
  return {{"time", std::string(unit)}};
}

inline nlohmann::json AngleUnit(std::string_view unit) {
  return {{"angle", std::string(unit)}};
}

inline nlohmann::json DensityUnit(std::string_view unit) {
  return {{"density", std::string(unit)}};
}

inline nlohmann::json SpeedUnit(std::string_view unit) {
  return {{"speed", std::string(unit)}};
}

inline nlohmann::json FrequencyUnit(std::string_view unit) {
  return {{"frequency", std::string(unit)}};
}

inline nlohmann::json VoltageUnit(std::string_view unit) {
  return {{"voltage", std::string(unit)}};
}

/**
 * @brief True when `unit` looks like an MDIO v1 unit model object.
 */
inline bool IsUnitModel(const nlohmann::json& unit) {
  if (!unit.is_object() || unit.size() != 1) {
    return false;
  }
  const std::string& key = unit.begin().key();
  return key == "length" || key == "time" || key == "angle" ||
         key == "density" || key == "speed" || key == "frequency" ||
         key == "voltage";
}

/**
 * @brief Wraps a unit model as Variable/Coordinate metadata (`unitsV1`).
 */
inline nlohmann::json UnitsMetadata(const nlohmann::json& unit) {
  return {{"unitsV1", unit}};
}

inline std::optional<nlohmann::json> UnitsMetadataOrNull(
    const std::optional<nlohmann::json>& unit) {
  if (!unit.has_value()) {
    return std::nullopt;
  }
  return UnitsMetadata(*unit);
}

/**
 * @brief Lowercases ASCII text. Used for domain / gather-domain parsing.
 */
inline std::string AsciiLower(std::string_view value) {
  return absl::AsciiStrToLower(value);
}

/**
 * @brief Python `str.capitalize()`: first char upper, rest lower.
 */
inline std::string AsciiCapitalize(std::string_view value) {
  std::string out = AsciiLower(value);
  if (!out.empty()) {
    out[0] =
        static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
  }
  return out;
}

}  // namespace builder
}  // namespace mdio

#endif  // MDIO_BUILDER_SCHEMAS_H_
