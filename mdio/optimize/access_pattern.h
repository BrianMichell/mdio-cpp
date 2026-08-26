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

#ifndef MDIO_OPTIMIZE_ACCESS_PATTERN_H_
#define MDIO_OPTIMIZE_ACCESS_PATTERN_H_

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "mdio/dataset.h"
#include "mdio/dataset_factory.h"

namespace mdio {
namespace optimize {

/**
 * @brief Configuration for fast-access copies and optional downsample mags.
 *
 * Mirrors mdio-python `OptimizedAccessPatternConfig`:
 *   - `optimize_dimensions` writes `fast_{dim}` siblings with the same shape
 *     and the same dimension names, only chunks + compressor change.
 *   - Python defaults to lossy ZFP. This TensorStore build does not ship ZFP,
 *     so the default here is lossless blosc/zstd.
 *
 * Extra (not in Python): `magnification_factors` writes `fast_mag_{n}`
 * downsample arrays. XY uses a box mean (NaNs skipped); Z/time is kept.
 * Downsampled axes get new names `{dim}_mag{n}` so xarray never sees the
 * same dim name at two sizes. WEBKNOSSOS already skips `fast_*` as layout
 * variants, so these do not become extra volumes.
 */
struct OptimizedAccessPatternConfig {
  /// Dim name -> on-disk chunk shape (same rank as the source variable).
  std::map<std::string, std::vector<Index>> optimize_dimensions;
  /// Optional tile sizes for the copy. Empty uses 128-voxel tiles.
  std::map<std::string, Index> processing_chunks;
  /// Optional downsample factors (2, 4, 8, ...). Writes `fast_mag_{n}`.
  std::vector<int> magnification_factors;
  /// Empty = resolve `attributes.defaultVariableName` or a preferred volume.
  std::string source_variable;
  /// Blosc codec name. Only blosc is supported by the write path.
  std::string compressor_name = "blosc";
  /// Blosc cname (`zstd`, `lz4`, ...).
  std::string compressor_algorithm = "zstd";
  /// Blosc clevel in [0, 9].
  int compressor_level = 5;
  /// Cap in-flight tile bytes. 0 uses 2048 MiB.
  size_t max_memory_bytes = 2048ull * 1024ull * 1024ull;
  /// Worker threads for tile copies. 0 uses hardware_concurrency.
  int max_threads = 8;
  /// Optional progress: name (`fast_mag_2`), tiles done, tiles total.
  std::function<void(const std::string&, int64_t, int64_t)> progress;
};

namespace internal {

inline std::string NormalizeName(std::string name) {
  absl::AsciiStrToLower(&name);
  std::replace(name.begin(), name.end(), ' ', '-');
  return name;
}

inline bool IsStoredMagName(const std::string& name) {
  const std::string n = NormalizeName(name);
  return n.rfind("fast_mag_", 0) == 0 || n.rfind("fast-mag-", 0) == 0;
}

inline int MagFactorFromStoredName(const std::string& name) {
  const std::string n = NormalizeName(name);
  auto parse_after = [&](const std::string& prefix) -> int {
    if (n.rfind(prefix, 0) != 0) {
      return 0;
    }
    try {
      const int factor = std::stoi(n.substr(prefix.size()));
      return factor >= 2 ? factor : 0;
    } catch (...) {
      return 0;
    }
  };
  const int underscored = parse_after("fast_mag_");
  if (underscored >= 2) {
    return underscored;
  }
  return parse_after("fast-mag-");
}

inline bool IsLayoutVariantName(const std::string& name) {
  const std::string n = NormalizeName(name);
  return n.rfind("fast-", 0) == 0 || n.rfind("fast_", 0) == 0 || n == "headers" ||
         n == "segy-file-header" || n == "segy_file_header" ||
         n == "image_headers";
}

// FAP siblings (`fast_crossline`, …) often use numcodecs.zfpy. This
// TensorStore build cannot open that codec. Keep `fast_mag_*` (blosc).
inline bool ShouldSkipUnopenableLayout(const std::string& name) {
  return IsLayoutVariantName(name) && !IsStoredMagName(name);
}

inline std::string VariableNameFromSpec(const nlohmann::json& spec) {
  if (spec.contains("name") && spec["name"].is_string()) {
    return spec["name"].get<std::string>();
  }
  std::string path;
  if (spec.contains("kvstore") && spec["kvstore"].is_object() &&
      spec["kvstore"].contains("path") && spec["kvstore"]["path"].is_string()) {
    path = spec["kvstore"]["path"].get<std::string>();
  }
  while (!path.empty() && (path.back() == '/' || path.back() == '\\')) {
    path.pop_back();
  }
  const auto slash = path.find_last_of("/\\");
  if (slash == std::string::npos) {
    return path;
  }
  return path.substr(slash + 1);
}

inline bool IsCoordinateLikeName(const std::string& name) {
  const std::string n = NormalizeName(name);
  static const char* kAliases[] = {
      "inline",  "iline", "crossline", "xline", "cdp-x", "cdp_x", "cdpx",
      "cdp-y",   "cdp_y", "cdpy",      "x",     "y",     "sample", "depth",
      "time",    "z",     "twt",       "tvd",   "tvdss", "offset", "cdp",
      "channel", "frequency", "angle", "gather", "trace", "label"};
  for (const char* alias : kAliases) {
    if (n == alias) {
      return true;
    }
  }
  return false;
}

inline bool IsDepthLikeName(const std::string& name) {
  const std::string n = NormalizeName(name);
  return n == "sample" || n == "depth" || n == "time" || n == "z" || n == "twt" ||
         n == "tvd" || n == "tvdss";
}

inline bool IsPreferredVolumeName(const std::string& name) {
  const std::string n = NormalizeName(name);
  return n == "amplitude" || n == "image" || n == "seismic" || n == "data" ||
         n == "grid" || n == "velocity" || n == "stack";
}

inline Result<std::string> DataTypeToMdioName(DataType dt) {
  if (dt == constants::kBool) return std::string("bool");
  if (dt == constants::kInt8) return std::string("int8");
  if (dt == constants::kInt16) return std::string("int16");
  if (dt == constants::kInt32) return std::string("int32");
  if (dt == constants::kInt64) return std::string("int64");
  if (dt == constants::kUint8) return std::string("uint8");
  if (dt == constants::kUint16) return std::string("uint16");
  if (dt == constants::kUint32) return std::string("uint32");
  if (dt == constants::kUint64) return std::string("uint64");
  if (dt == constants::kFloat16) return std::string("float16");
  if (dt == constants::kFloat32) return std::string("float32");
  if (dt == constants::kFloat64) return std::string("float64");
  if (dt == constants::kComplex64) return std::string("complex64");
  if (dt == constants::kComplex128) return std::string("complex128");
  return absl::InvalidArgumentError(
      "Unsupported dtype for access-pattern generation");
}

inline nlohmann::json LosslessCompressor(
    const OptimizedAccessPatternConfig& config) {
  if (config.compressor_name != "blosc") {
    return nlohmann::json{
        {"name", "blosc"},
        {"algorithm", config.compressor_algorithm},
        {"level", config.compressor_level}};
  }
  return nlohmann::json{{"name", config.compressor_name},
                        {"algorithm", config.compressor_algorithm},
                        {"level", config.compressor_level}};
}

inline std::vector<std::string> DimensionNames(const Variable<>& var) {
  std::vector<std::string> names;
  for (const auto& label : var.dimensions().labels()) {
    if (!label.empty()) {
      names.emplace_back(label);
    }
  }
  return names;
}

inline std::vector<Index> DimensionSizes(const Variable<>& var) {
  std::vector<Index> sizes;
  const auto domain = var.dimensions();
  for (DimensionIndex i = 0; i < domain.rank(); ++i) {
    if (!domain.labels()[i].empty()) {
      sizes.push_back(domain.shape()[i]);
    }
  }
  return sizes;
}

inline Result<zarr::ZarrVersion> DetectVariableVersion(const Variable<>& var) {
  MDIO_ASSIGN_OR_RETURN(auto spec, var.spec());
  MDIO_ASSIGN_OR_RETURN(auto json, spec.ToJson(IncludeDefaults{}));
  return zarr::GetVersionFromSpec(json);
}

inline bool LooksStructured(const Variable<>& var) {
  auto spec = var.spec();
  if (!spec.ok()) {
    return false;
  }
  auto json = spec->ToJson(IncludeDefaults{});
  if (!json.ok() || !json->contains("metadata")) {
    return false;
  }
  return zarr::IsStructuredDType(zarr::GetVersionFromSpec(json.value()),
                                 json.value()["metadata"]);
}

inline Result<std::string> ResolveSourceVariable(
    const Dataset& dataset, const std::string& requested) {
  if (!requested.empty()) {
    if (!dataset.variables.contains_key(requested)) {
      return absl::NotFoundError("Source variable '" + requested +
                                 "' not found.");
    }
    return requested;
  }

  const auto& meta = dataset.getMetadata();
  if (meta.contains("attributes") &&
      meta["attributes"].contains("defaultVariableName") &&
      meta["attributes"]["defaultVariableName"].is_string()) {
    const std::string name =
        meta["attributes"]["defaultVariableName"].get<std::string>();
    if (!dataset.variables.contains_key(name)) {
      return absl::NotFoundError("defaultVariableName '" + name +
                                 "' not found.");
    }
    return name;
  }

  for (const auto& key : dataset.variables.get_iterable_accessor()) {
    if (IsPreferredVolumeName(key) && !IsLayoutVariantName(key)) {
      return key;
    }
  }

  for (const auto& key : dataset.variables.get_iterable_accessor()) {
    if (IsLayoutVariantName(key) || IsCoordinateLikeName(key)) {
      continue;
    }
    MDIO_ASSIGN_OR_RETURN(auto var, dataset.variables.at(key));
    if (LooksStructured(var)) {
      continue;
    }
    if (DimensionNames(var).size() >= 2) {
      return key;
    }
  }
  return absl::NotFoundError(
      "No volume variable found for access-pattern generation.");
}

inline nlohmann::json BuildVariableSchema(
    const std::string& name, const std::string& dtype,
    const std::vector<std::pair<std::string, uint64_t>>& dims,
    const std::vector<int64_t>& chunks,
    const std::vector<std::string>& coordinates, const std::string& long_name,
    const nlohmann::json& compressor) {
  nlohmann::json dimensions = nlohmann::json::array();
  for (const auto& [dim_name, size] : dims) {
    dimensions.push_back({{"name", dim_name}, {"size", size}});
  }
  nlohmann::json schema = {{"name", name},
                           {"dataType", dtype},
                           {"dimensions", dimensions},
                           {"longName", long_name},
                           {"metadata",
                            {{"chunkGrid",
                              {{"name", "regular"},
                               {"configuration", {{"chunkShape", chunks}}}}}}}};
  if (!coordinates.empty()) {
    schema["coordinates"] = coordinates;
  }
  if (!compressor.is_null() && !compressor.empty()) {
    schema["compressor"] = compressor;
  }
  return schema;
}

inline Result<Variable<>> CreateVariableFromSchema(
    nlohmann::json schema, const std::string& dataset_path,
    zarr::ZarrVersion version,
    std::unordered_map<std::string, uint64_t> dimension_map,
    tensorstore::Context context) {
  MDIO_ASSIGN_OR_RETURN(
      auto spec, ::from_json_to_spec(schema, dimension_map, dataset_path,
                                     version));
  return Variable<>::Open(spec, constants::kCreateClean, context).result();
}

inline Index CeilDiv(Index numerator, int denominator) {
  return (numerator + static_cast<Index>(denominator) - 1) /
         static_cast<Index>(denominator);
}

inline bool ShapeEquals(const Variable<>& var, const std::vector<Index>& sizes) {
  const auto domain = var.dimensions();
  if (domain.rank() != static_cast<DimensionIndex>(sizes.size())) {
    return false;
  }
  for (DimensionIndex i = 0; i < domain.rank(); ++i) {
    if (domain.shape()[i] != sizes[static_cast<size_t>(i)]) {
      return false;
    }
  }
  return true;
}

inline Index TileSizeForDim(const std::map<std::string, Index>& processing,
                            const std::string& dim, Index dim_size) {
  const auto it = processing.find(dim);
  if (it != processing.end() && it->second > 0) {
    return std::min(it->second, dim_size);
  }
  return std::min<Index>(dim_size, 128);
}

inline int64_t SafeProduct(const std::vector<Index>& values) {
  int64_t product = 1;
  for (Index value : values) {
    if (value <= 0) {
      return 0;
    }
    if (product > std::numeric_limits<int64_t>::max() / value) {
      return std::numeric_limits<int64_t>::max() / 2;
    }
    product *= value;
  }
  return product;
}

inline int64_t TileMemoryBytes(const std::vector<Index>& dest_extents,
                               const std::vector<Index>& src_steps,
                               Index dtype_bytes) {
  std::vector<Index> src_extents = dest_extents;
  for (size_t i = 0; i < src_extents.size() && i < src_steps.size(); ++i) {
    const Index step = std::max<Index>(src_steps[i], 1);
    if (src_extents[i] > std::numeric_limits<Index>::max() / step) {
      src_extents[i] = std::numeric_limits<Index>::max();
    } else {
      src_extents[i] *= step;
    }
  }
  const int64_t dest_voxels = SafeProduct(dest_extents);
  const int64_t src_voxels = SafeProduct(src_extents);
  const int64_t elem = std::max<Index>(dtype_bytes, 1);
  if (src_voxels > (std::numeric_limits<int64_t>::max() / elem) - dest_voxels) {
    return std::numeric_limits<int64_t>::max() / 2;
  }
  return (src_voxels + dest_voxels) * elem;
}

inline int EffectiveThreadCount(const OptimizedAccessPatternConfig& config) {
  if (config.max_threads > 0) {
    return config.max_threads;
  }
  const unsigned hardware = std::thread::hardware_concurrency();
  return hardware == 0 ? 1 : static_cast<int>(hardware);
}

inline size_t EffectiveMemoryBudget(const OptimizedAccessPatternConfig& config) {
  return config.max_memory_bytes == 0 ? 2048ull * 1024ull * 1024ull
                                      : config.max_memory_bytes;
}

struct TileWork {
  std::vector<std::string> src_labels;
  std::vector<std::string> dst_labels;
  std::vector<Index> starts;
  std::vector<Index> stops;
  std::vector<Index> src_starts;
  std::vector<Index> src_stops;
  std::vector<Index> steps;
  int64_t memory_bytes;
};

inline Result<void> CopyOneTile(const Variable<>& source, const Variable<>& dest,
                                const TileWork& tile) {
  std::vector<RangeDescriptor<Index>> src_slices;
  std::vector<RangeDescriptor<Index>> dst_slices;
  src_slices.reserve(tile.src_labels.size());
  dst_slices.reserve(tile.dst_labels.size());
  for (size_t i = 0; i < tile.src_labels.size(); ++i) {
    src_slices.push_back(
        {tile.src_labels[i], tile.starts[i], tile.stops[i], 1});
    dst_slices.push_back(
        {tile.dst_labels[i], tile.starts[i], tile.stops[i], 1});
  }
  Variable<> src_handle = source;
  Variable<> dst_handle = dest;
  MDIO_ASSIGN_OR_RETURN(auto src_tile, src_handle.slice(src_slices));
  MDIO_ASSIGN_OR_RETURN(auto dst_tile, dst_handle.slice(dst_slices));
  auto read = src_tile.Read().result();
  if (!read.status().ok()) {
    return read.status();
  }
  auto write = dst_tile.Write(read.value());
  if (!write.status().ok()) {
    return write.status();
  }
  return absl::OkStatus();
}

inline int64_t COrderIndex(const std::vector<Index>& coords,
                           const std::vector<Index>& shape) {
  int64_t index = 0;
  for (size_t i = 0; i < coords.size(); ++i) {
    index = index * shape[i] + coords[i];
  }
  return index;
}

template <typename T>
void AverageBoxInto(const T* src, T* dest, const std::vector<Index>& src_shape,
                    const std::vector<Index>& dest_shape,
                    const std::vector<Index>& steps) {
  const size_t rank = dest_shape.size();
  std::vector<Index> dest_coord(rank, 0);
  while (true) {
    double sum = 0.0;
    int count = 0;
    std::vector<Index> src_coord(rank, 0);
    std::vector<Index> src_begin(rank, 0);
    std::vector<Index> src_end(rank, 0);
    for (size_t i = 0; i < rank; ++i) {
      const Index step = i < steps.size() ? std::max<Index>(steps[i], 1) : 1;
      src_begin[i] = dest_coord[i] * step;
      src_end[i] = std::min(src_begin[i] + step, src_shape[i]);
      src_coord[i] = src_begin[i];
    }
    while (true) {
      const T sample = src[COrderIndex(src_coord, src_shape)];
      if constexpr (std::is_floating_point_v<T>) {
        if (!std::isnan(static_cast<double>(sample))) {
          sum += static_cast<double>(sample);
          ++count;
        }
      } else {
        sum += static_cast<double>(sample);
        ++count;
      }
      int dim = static_cast<int>(rank) - 1;
      while (dim >= 0) {
        ++src_coord[static_cast<size_t>(dim)];
        if (src_coord[static_cast<size_t>(dim)] <
            src_end[static_cast<size_t>(dim)]) {
          break;
        }
        src_coord[static_cast<size_t>(dim)] = src_begin[static_cast<size_t>(dim)];
        --dim;
      }
      if (dim < 0) {
        break;
      }
    }
    if (count > 0) {
      if constexpr (std::is_floating_point_v<T>) {
        dest[COrderIndex(dest_coord, dest_shape)] = static_cast<T>(sum / count);
      } else {
        dest[COrderIndex(dest_coord, dest_shape)] =
            static_cast<T>(std::llround(sum / count));
      }
    }
    int dim = static_cast<int>(rank) - 1;
    while (dim >= 0) {
      ++dest_coord[static_cast<size_t>(dim)];
      if (dest_coord[static_cast<size_t>(dim)] <
          dest_shape[static_cast<size_t>(dim)]) {
        break;
      }
      dest_coord[static_cast<size_t>(dim)] = 0;
      --dim;
    }
    if (dim < 0) {
      break;
    }
  }
}

inline Result<void> AverageOneTile(const Variable<>& source,
                                   const Variable<>& dest,
                                   const TileWork& tile) {
  std::vector<RangeDescriptor<Index>> src_slices;
  std::vector<RangeDescriptor<Index>> dst_slices;
  src_slices.reserve(tile.src_labels.size());
  dst_slices.reserve(tile.dst_labels.size());
  for (size_t i = 0; i < tile.src_labels.size(); ++i) {
    src_slices.push_back(
        {tile.src_labels[i], tile.src_starts[i], tile.src_stops[i], 1});
    dst_slices.push_back(
        {tile.dst_labels[i], tile.starts[i], tile.stops[i], 1});
  }
  Variable<> src_handle = source;
  Variable<> dst_handle = dest;
  MDIO_ASSIGN_OR_RETURN(auto src_tile, src_handle.slice(src_slices));
  MDIO_ASSIGN_OR_RETURN(auto dst_tile, dst_handle.slice(dst_slices));
  auto read = src_tile.Read().result();
  if (!read.status().ok()) {
    return read.status();
  }
  MDIO_ASSIGN_OR_RETURN(auto dest_data, from_variable(dst_tile));
  const auto src_domain = src_tile.dimensions();
  const auto dst_domain = dst_tile.dimensions();
  std::vector<Index> src_shape;
  std::vector<Index> dest_shape;
  src_shape.reserve(static_cast<size_t>(src_domain.rank()));
  dest_shape.reserve(static_cast<size_t>(dst_domain.rank()));
  for (DimensionIndex i = 0; i < src_domain.rank(); ++i) {
    src_shape.push_back(src_domain.shape()[i]);
    dest_shape.push_back(dst_domain.shape()[i]);
  }
  auto src_ptr = read.value().get_data_accessor().byte_strided_origin_pointer();
  auto dst_ptr = dest_data.get_data_accessor().byte_strided_origin_pointer();
  const DataType dtype = source.dtype();
  if (dtype == constants::kFloat32) {
    AverageBoxInto(static_cast<const float*>(src_ptr.get()),
                   static_cast<float*>(dst_ptr.get()), src_shape, dest_shape,
                   tile.steps);
  } else if (dtype == constants::kFloat64) {
    AverageBoxInto(static_cast<const double*>(src_ptr.get()),
                   static_cast<double*>(dst_ptr.get()), src_shape, dest_shape,
                   tile.steps);
  } else if (dtype == constants::kInt32) {
    AverageBoxInto(static_cast<const int32_t*>(src_ptr.get()),
                   static_cast<int32_t*>(dst_ptr.get()), src_shape, dest_shape,
                   tile.steps);
  } else if (dtype == constants::kUint32) {
    AverageBoxInto(static_cast<const uint32_t*>(src_ptr.get()),
                   static_cast<uint32_t*>(dst_ptr.get()), src_shape, dest_shape,
                   tile.steps);
  } else if (dtype == constants::kInt16) {
    AverageBoxInto(static_cast<const int16_t*>(src_ptr.get()),
                   static_cast<int16_t*>(dst_ptr.get()), src_shape, dest_shape,
                   tile.steps);
  } else if (dtype == constants::kUint16) {
    AverageBoxInto(static_cast<const uint16_t*>(src_ptr.get()),
                   static_cast<uint16_t*>(dst_ptr.get()), src_shape, dest_shape,
                   tile.steps);
  } else if (dtype == constants::kInt8) {
    AverageBoxInto(static_cast<const int8_t*>(src_ptr.get()),
                   static_cast<int8_t*>(dst_ptr.get()), src_shape, dest_shape,
                   tile.steps);
  } else if (dtype == constants::kUint8) {
    AverageBoxInto(static_cast<const uint8_t*>(src_ptr.get()),
                   static_cast<uint8_t*>(dst_ptr.get()), src_shape, dest_shape,
                   tile.steps);
  } else {
    return absl::UnimplementedError(
        "Box-average downsample is not implemented for this dtype.");
  }
  auto write = dst_tile.Write(dest_data);
  if (!write.status().ok()) {
    return write.status();
  }
  return absl::OkStatus();
}

inline void LogOversizedTile(int64_t bytes, int64_t budget) {
  if (budget <= 0 || budget == std::numeric_limits<int64_t>::max() ||
      bytes <= budget) {
    return;
  }
  std::cerr << "mdio_optimize: tile needs " << bytes
            << " bytes, over budget " << budget << "; running anyway\n";
}

inline Result<void> RunTiles(
    const Variable<>& source, const Variable<>& dest,
    const std::vector<TileWork>& tiles, int max_threads, size_t max_memory_bytes,
    const std::string& progress_name,
    const std::function<void(const std::string&, int64_t, int64_t)>& progress,
    bool average = false) {
  if (tiles.empty()) {
    return absl::OkStatus();
  }
  const int64_t total = static_cast<int64_t>(tiles.size());
  const int threads = std::max(1, max_threads);
  const int64_t budget = max_memory_bytes == 0
                             ? std::numeric_limits<int64_t>::max()
                             : static_cast<int64_t>(max_memory_bytes);

  if (threads == 1 || tiles.size() == 1) {
    int64_t done = 0;
    for (const auto& tile : tiles) {
      LogOversizedTile(tile.memory_bytes, budget);
      auto copied = average ? AverageOneTile(source, dest, tile)
                            : CopyOneTile(source, dest, tile);
      if (!copied.ok()) {
        return copied.status();
      }
      ++done;
      if (progress) {
        progress(progress_name, done, total);
      }
    }
    return absl::OkStatus();
  }

  std::mutex mu;
  std::condition_variable cv;
  size_t next = 0;
  int64_t in_flight = 0;
  int64_t done = 0;
  absl::Status error;

  auto worker = [&]() {
    while (true) {
      TileWork tile;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] {
          if (!error.ok()) {
            return true;
          }
          if (next >= tiles.size()) {
            return true;
          }
          return in_flight == 0 ||
                 in_flight + tiles[next].memory_bytes <= budget;
        });
        if (!error.ok() || next >= tiles.size()) {
          return;
        }
        tile = tiles[next++];
        in_flight += tile.memory_bytes;
        LogOversizedTile(tile.memory_bytes, budget);
      }
      auto copied = average ? AverageOneTile(source, dest, tile)
                            : CopyOneTile(source, dest, tile);
      {
        std::lock_guard<std::mutex> lock(mu);
        in_flight -= tile.memory_bytes;
        if (!copied.ok() && error.ok()) {
          error = copied.status();
        } else if (copied.ok()) {
          ++done;
          if (progress) {
            progress(progress_name, done, total);
          }
        }
        cv.notify_all();
      }
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(threads));
  for (int i = 0; i < threads; ++i) {
    pool.emplace_back(worker);
  }
  for (auto& thread : pool) {
    thread.join();
  }
  if (!error.ok()) {
    return error;
  }
  return absl::OkStatus();
}

inline Result<void> CopyAlignedTiled(
    const Variable<>& source, const Variable<>& dest,
    const OptimizedAccessPatternConfig& config,
    const std::string& progress_name = {},
    const std::vector<Index>& src_steps = {}) {
  const auto src_domain = source.dimensions();
  const auto dst_domain = dest.dimensions();
  if (src_domain.rank() != dst_domain.rank()) {
    return absl::InvalidArgumentError(
        "Source and destination ranks differ during copy.");
  }
  const DimensionIndex rank = src_domain.rank();
  std::vector<std::string> src_labels;
  std::vector<std::string> dst_labels;
  std::vector<Index> sizes;
  std::vector<Index> tiles;
  src_labels.reserve(static_cast<size_t>(rank));
  dst_labels.reserve(static_cast<size_t>(rank));
  for (DimensionIndex i = 0; i < rank; ++i) {
    if (src_domain.labels()[i].empty() || dst_domain.labels()[i].empty()) {
      return absl::InvalidArgumentError(
          "Cannot copy a structured / unlabeled dimension.");
    }
    if (src_domain.shape()[i] != dst_domain.shape()[i]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Source and destination shapes differ during copy: dim ", i, " src=",
          src_domain.shape()[i], " (", src_domain.labels()[i], ") dst=",
          dst_domain.shape()[i], " (", dst_domain.labels()[i], ")"));
    }
    src_labels.emplace_back(src_domain.labels()[i]);
    dst_labels.emplace_back(dst_domain.labels()[i]);
    sizes.push_back(src_domain.shape()[i]);
    tiles.push_back(TileSizeForDim(config.processing_chunks, src_labels.back(),
                                   sizes.back()));
  }

  const Index dtype_bytes = source.dtype().size();
  std::vector<TileWork> work;
  std::vector<Index> cursor(static_cast<size_t>(rank), 0);
  while (true) {
    TileWork tile;
    tile.src_labels = src_labels;
    tile.dst_labels = dst_labels;
    std::vector<Index> extents;
    extents.reserve(static_cast<size_t>(rank));
    for (DimensionIndex i = 0; i < rank; ++i) {
      const Index origin = src_domain.origin()[i];
      const Index begin = origin + cursor[static_cast<size_t>(i)];
      const Index end = std::min(begin + tiles[static_cast<size_t>(i)],
                                 origin + sizes[static_cast<size_t>(i)]);
      tile.starts.push_back(begin);
      tile.stops.push_back(end);
      extents.push_back(end - begin);
    }
    tile.memory_bytes = TileMemoryBytes(extents, src_steps, dtype_bytes);
    work.push_back(std::move(tile));

    int dim = static_cast<int>(rank) - 1;
    while (dim >= 0) {
      cursor[static_cast<size_t>(dim)] += tiles[static_cast<size_t>(dim)];
      if (cursor[static_cast<size_t>(dim)] < sizes[static_cast<size_t>(dim)]) {
        break;
      }
      cursor[static_cast<size_t>(dim)] = 0;
      --dim;
    }
    if (dim < 0) {
      break;
    }
  }

  return RunTiles(source, dest, work, EffectiveThreadCount(config),
                  EffectiveMemoryBudget(config), progress_name, config.progress);
}

inline Result<void> CopyDownsampledTiled(
    const Variable<>& source, const Variable<>& dest,
    const OptimizedAccessPatternConfig& config,
    const std::string& progress_name, const std::vector<Index>& steps) {
  const auto src_domain = source.dimensions();
  const auto dst_domain = dest.dimensions();
  if (src_domain.rank() != dst_domain.rank()) {
    return absl::InvalidArgumentError(
        "Source and destination ranks differ during downsample.");
  }
  const DimensionIndex rank = src_domain.rank();
  std::vector<std::string> src_labels;
  std::vector<std::string> dst_labels;
  std::vector<Index> dest_sizes;
  std::vector<Index> tiles;
  src_labels.reserve(static_cast<size_t>(rank));
  dst_labels.reserve(static_cast<size_t>(rank));
  for (DimensionIndex i = 0; i < rank; ++i) {
    if (src_domain.labels()[i].empty() || dst_domain.labels()[i].empty()) {
      return absl::InvalidArgumentError(
          "Cannot downsample a structured / unlabeled dimension.");
    }
    src_labels.emplace_back(src_domain.labels()[i]);
    dst_labels.emplace_back(dst_domain.labels()[i]);
    dest_sizes.push_back(dst_domain.shape()[i]);
    tiles.push_back(TileSizeForDim(config.processing_chunks, dst_labels.back(),
                                   dest_sizes.back()));
  }

  const Index dtype_bytes = source.dtype().size();
  std::vector<TileWork> work;
  std::vector<Index> cursor(static_cast<size_t>(rank), 0);
  while (true) {
    TileWork tile;
    tile.src_labels = src_labels;
    tile.dst_labels = dst_labels;
    tile.steps = steps;
    std::vector<Index> dest_extents;
    dest_extents.reserve(static_cast<size_t>(rank));
    for (DimensionIndex i = 0; i < rank; ++i) {
      const Index dest_origin = dst_domain.origin()[i];
      const Index src_origin = src_domain.origin()[i];
      const Index dest_begin = dest_origin + cursor[static_cast<size_t>(i)];
      const Index dest_end =
          std::min(dest_begin + tiles[static_cast<size_t>(i)],
                   dest_origin + dest_sizes[static_cast<size_t>(i)]);
      const Index step =
          i < static_cast<DimensionIndex>(steps.size()) && steps[i] > 0
              ? steps[i]
              : 1;
      const Index src_begin =
          src_origin + (dest_begin - dest_origin) * step;
      const Index src_end = std::min(
          src_origin + src_domain.shape()[i],
          src_origin + (dest_end - dest_origin) * step);
      tile.starts.push_back(dest_begin);
      tile.stops.push_back(dest_end);
      tile.src_starts.push_back(src_begin);
      tile.src_stops.push_back(src_end);
      dest_extents.push_back(dest_end - dest_begin);
    }
    tile.memory_bytes = TileMemoryBytes(dest_extents, steps, dtype_bytes);
    work.push_back(std::move(tile));

    int dim = static_cast<int>(rank) - 1;
    while (dim >= 0) {
      cursor[static_cast<size_t>(dim)] += tiles[static_cast<size_t>(dim)];
      if (cursor[static_cast<size_t>(dim)] < dest_sizes[static_cast<size_t>(dim)]) {
        break;
      }
      cursor[static_cast<size_t>(dim)] = 0;
      --dim;
    }
    if (dim < 0) {
      break;
    }
  }

  return RunTiles(source, dest, work, EffectiveThreadCount(config),
                  EffectiveMemoryBudget(config), progress_name, config.progress,
                  /*average=*/true);
}

inline void RegisterVariable(Dataset& dataset, Variable<> var,
                             const std::vector<std::string>& coordinates) {
  const std::string name = var.get_variable_name();
  var.set_metadata_publish_flag(true);
  dataset.variables.add(name, var);
  if (!coordinates.empty()) {
    dataset.coordinates[name] = coordinates;
  }
}

inline Result<void> RebuildDomain(Dataset& dataset) {
  std::unordered_map<std::string, Index> shape_size;
  for (const auto& key : dataset.variables.get_iterable_accessor()) {
    MDIO_ASSIGN_OR_RETURN(auto var, dataset.variables.at(key));
    const auto domain = var.dimensions();
    auto shape = domain.shape().cbegin();
    for (const auto& label : domain.labels()) {
      if (!label.empty()) {
        const std::string name(label);
        if (shape_size.count(name) && shape_size[name] != *shape) {
          return absl::InvalidArgumentError("Dimension '" + name +
                                            "' has conflicting sizes");
        }
        shape_size[name] = *shape;
      }
      ++shape;
    }
  }

  std::vector<std::string> keys;
  std::vector<Index> values;
  keys.reserve(shape_size.size());
  values.reserve(shape_size.size());
  for (const auto& [key, value] : shape_size) {
    keys.push_back(key);
    values.push_back(value);
  }
  MDIO_ASSIGN_OR_RETURN(auto domain, tensorstore::IndexDomainBuilder<>(
                                         static_cast<DimensionIndex>(
                                             shape_size.size()))
                                         .shape(values)
                                         .labels(keys)
                                         .Finalize());
  dataset.domain = domain;
  return absl::OkStatus();
}

inline Result<void> WriteFastAccessVariable(
    Dataset& dataset, const Variable<>& source, const std::string& dataset_path,
    zarr::ZarrVersion version, const std::string& dim_name,
    const std::vector<Index>& chunks,
    const OptimizedAccessPatternConfig& config) {
  const std::string dest_name = "fast_" + dim_name;
  if (dataset.variables.contains_key(dest_name)) {
    return absl::AlreadyExistsError(
        "Access-pattern variable '" + dest_name +
        "' already exists. Delete it before regenerating.");
  }

  const auto dim_names = DimensionNames(source);
  const auto dim_sizes = DimensionSizes(source);
  if (dim_names.size() != static_cast<size_t>(chunks.size())) {
    return absl::InvalidArgumentError(
        "Chunk shape for '" + dim_name + "' must have rank " +
        std::to_string(dim_names.size()) + ", got " +
        std::to_string(chunks.size()));
  }
  if (std::find(dim_names.begin(), dim_names.end(), dim_name) ==
      dim_names.end()) {
    return absl::InvalidArgumentError("Dimension to optimize '" + dim_name +
                                      "' not found in source dims.");
  }

  MDIO_ASSIGN_OR_RETURN(const std::string dtype,
                        DataTypeToMdioName(source.dtype()));

  std::vector<std::pair<std::string, uint64_t>> dims;
  std::unordered_map<std::string, uint64_t> dimension_map;
  dims.reserve(dim_names.size());
  for (size_t i = 0; i < dim_names.size(); ++i) {
    dims.emplace_back(dim_names[i], static_cast<uint64_t>(dim_sizes[i]));
    dimension_map[dim_names[i]] = static_cast<uint64_t>(dim_sizes[i]);
  }

  std::vector<std::string> coordinates = dim_names;
  if (dataset.coordinates.count(source.get_variable_name())) {
    coordinates = dataset.coordinates.at(source.get_variable_name());
  }

  std::vector<int64_t> chunk_i64(chunks.begin(), chunks.end());
  auto schema = BuildVariableSchema(
      dest_name, dtype, dims, chunk_i64, coordinates,
      dim_name + " optimized access pattern of " + source.get_variable_name(),
      LosslessCompressor(config));

  MDIO_ASSIGN_OR_RETURN(
      auto dest, CreateVariableFromSchema(std::move(schema), dataset_path,
                                          version, dimension_map,
                                          dataset.getContext()));
  auto copy = CopyAlignedTiled(source, dest, config, dest_name);
  if (!copy.ok()) {
    return copy.status();
  }
  RegisterVariable(dataset, dest, coordinates);
  return absl::OkStatus();
}

inline std::vector<bool> DimsToDownsample(const std::vector<std::string>& names) {
  std::vector<bool> downsample(names.size(), true);
  bool found_z = false;
  for (size_t i = 0; i < names.size(); ++i) {
    if (IsDepthLikeName(names[i])) {
      downsample[i] = false;
      found_z = true;
    }
  }
  if (!found_z && names.size() >= 3) {
    downsample.back() = false;
  }
  return downsample;
}

inline std::vector<Index> ExpectedMagShape(
    const std::vector<Index>& orig_sizes, const std::vector<bool>& downsample,
    int factor) {
  std::vector<Index> sizes = orig_sizes;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i < downsample.size() && downsample[i]) {
      sizes[i] = CeilDiv(orig_sizes[i], factor);
    }
  }
  return sizes;
}

// Largest stored mag P where F % P == 0 and shape matches ceil(N/P).
// No such P → original volume with factor 1 (full-res fallback).
inline Result<std::pair<Variable<>, int>> ResolveMagParent(
    Dataset& dataset, const Variable<>& original, int factor,
    const std::vector<bool>& downsample) {
  const auto orig_sizes = DimensionSizes(original);
  int best_p = 1;
  Variable<> best = original;
  for (const auto& key : dataset.variables.get_iterable_accessor()) {
    if (!IsStoredMagName(key)) {
      continue;
    }
    const int parent_factor = MagFactorFromStoredName(key);
    if (parent_factor < 2 || parent_factor >= factor ||
        factor % parent_factor != 0 || parent_factor <= best_p) {
      continue;
    }
    MDIO_ASSIGN_OR_RETURN(auto candidate, dataset.variables.at(key));
    if (!ShapeEquals(candidate, ExpectedMagShape(orig_sizes, downsample,
                                                 parent_factor))) {
      continue;
    }
    best = candidate;
    best_p = parent_factor;
  }
  return std::make_pair(best, best_p);
}

inline Result<bool> WriteMagnification(
    Dataset& dataset, const Variable<>& source, const std::string& dataset_path,
    zarr::ZarrVersion version, int factor,
    const OptimizedAccessPatternConfig& config) {
  if (factor < 2) {
    return absl::InvalidArgumentError(
        "magnification_factors entries must be >= 2");
  }
  const std::string dest_name = "fast_mag_" + std::to_string(factor);

  const auto src_names = DimensionNames(source);
  const auto src_sizes = DimensionSizes(source);
  if (src_names.size() < 2) {
    return absl::InvalidArgumentError(
        "Magnification requires a source rank of at least 2.");
  }
  const auto downsample = DimsToDownsample(src_names);
  MDIO_ASSIGN_OR_RETURN(auto parent_and_factor,
                        ResolveMagParent(dataset, source, factor, downsample));
  Variable<> parent = parent_and_factor.first;
  const int parent_factor = parent_and_factor.second;
  const int relative = factor / parent_factor;

  std::vector<std::string> dest_names;
  std::vector<Index> dest_sizes;
  std::vector<Index> steps;
  std::vector<Index> copy_steps;
  dest_names.reserve(src_names.size());
  for (size_t i = 0; i < src_names.size(); ++i) {
    if (downsample[i]) {
      // TensorStore half-open [0, N) step F has size ceil(N/F). Floor
      // drops the last sample and makes dest shorter than the sliced view.
      const Index dest_size = CeilDiv(src_sizes[i], factor);
      if (dest_size < 1) {
        return absl::InvalidArgumentError(
            "Dimension '" + src_names[i] + "' is smaller than mag factor " +
            std::to_string(factor));
      }
      dest_names.push_back(src_names[i] + "_mag" + std::to_string(factor));
      dest_sizes.push_back(dest_size);
      steps.push_back(factor);
      copy_steps.push_back(relative);
    } else {
      dest_names.push_back(src_names[i]);
      dest_sizes.push_back(src_sizes[i]);
      steps.push_back(1);
      copy_steps.push_back(1);
    }
  }

  std::vector<RangeDescriptor<Index>> src_slices;
  const auto src_domain = source.dimensions();
  for (size_t i = 0; i < src_names.size(); ++i) {
    if (steps[i] <= 1) {
      continue;
    }
    const Index start = src_domain.origin()[static_cast<DimensionIndex>(i)];
    const Index stop = start + src_sizes[i];
    src_slices.push_back({src_names[i], start, stop, steps[i]});
  }
  Variable<> src_view = source;
  if (!src_slices.empty()) {
    MDIO_ASSIGN_OR_RETURN(src_view, src_view.slice(src_slices));
  }
  const auto view_domain = src_view.dimensions();
  std::vector<std::pair<std::string, uint64_t>> dest_dims;
  std::unordered_map<std::string, uint64_t> dimension_map;
  for (size_t i = 0; i < dest_names.size(); ++i) {
    dest_sizes[i] = view_domain.shape()[static_cast<DimensionIndex>(i)];
    dest_dims.emplace_back(dest_names[i], static_cast<uint64_t>(dest_sizes[i]));
    dimension_map[dest_names[i]] = static_cast<uint64_t>(dest_sizes[i]);
  }

  bool wrote = false;
  // Dimension coordinates for new mag dims (xarray dim-coords). Rank-1, so
  // WEBKNOSSOS will not treat them as volumes. V3 Open lists leftover S3
  // arrays from a failed prior run; recreate if the size does not match
  // the dest volume.
  for (size_t i = 0; i < src_names.size(); ++i) {
    if (!downsample[i]) {
      continue;
    }
    const std::string& dest_dim = dest_names[i];
    if (dataset.variables.contains_key(dest_dim)) {
      MDIO_ASSIGN_OR_RETURN(auto existing, dataset.variables.at(dest_dim));
      if (ShapeEquals(existing, {dest_sizes[i]})) {
        continue;
      }
    }
    std::string coord_dtype = "uint32";
    Variable<> stepped;
    bool have_stepped = false;
    if (dataset.variables.contains_key(src_names[i])) {
      MDIO_ASSIGN_OR_RETURN(auto src_coord,
                            dataset.variables.at(src_names[i]));
      MDIO_ASSIGN_OR_RETURN(coord_dtype, DataTypeToMdioName(src_coord.dtype()));
      const auto coord_domain = src_coord.dimensions();
      const Index start = coord_domain.origin()[0];
      const Index stop = start + dest_sizes[i] * steps[i];
      std::vector<std::string> label_store = {src_names[i]};
      RangeDescriptor<Index> slice_desc{label_store[0], start, stop, steps[i]};
      MDIO_ASSIGN_OR_RETURN(stepped, src_coord.slice(slice_desc));
      have_stepped = true;
    }
    auto coord_schema = BuildVariableSchema(
        dest_dim, coord_dtype,
        {{dest_dim, static_cast<uint64_t>(dest_sizes[i])}},
        {static_cast<int64_t>(dest_sizes[i])}, {}, dest_dim,
        nlohmann::json());
    MDIO_ASSIGN_OR_RETURN(
        auto dest_coord,
        CreateVariableFromSchema(std::move(coord_schema), dataset_path, version,
                                 dimension_map, dataset.getContext()));
    if (have_stepped) {
      auto coord_copy = CopyAlignedTiled(stepped, dest_coord, config);
      if (!coord_copy.ok()) {
        return coord_copy.status();
      }
    }
    RegisterVariable(dataset, dest_coord, {});
    wrote = true;
  }

  if (dataset.variables.contains_key(dest_name)) {
    MDIO_ASSIGN_OR_RETURN(auto existing, dataset.variables.at(dest_name));
    if (ShapeEquals(existing, dest_sizes)) {
      return wrote;
    }
  }

  MDIO_ASSIGN_OR_RETURN(const std::string dtype,
                        DataTypeToMdioName(source.dtype()));
  std::vector<int64_t> chunks;
  chunks.reserve(dest_sizes.size());
  for (Index size : dest_sizes) {
    chunks.push_back(static_cast<int64_t>(std::min<Index>(size, 128)));
  }

  auto schema = BuildVariableSchema(
      dest_name, dtype, dest_dims, chunks, dest_names,
      "downsample magnification " + std::to_string(factor) + " of " +
          source.get_variable_name(),
      LosslessCompressor(config));
  MDIO_ASSIGN_OR_RETURN(
      auto dest, CreateVariableFromSchema(std::move(schema), dataset_path,
                                          version, dimension_map,
                                          dataset.getContext()));
  auto mag_copy =
      CopyDownsampledTiled(parent, dest, config, dest_name, copy_steps);
  if (!mag_copy.ok()) {
    return mag_copy.status();
  }
  RegisterVariable(dataset, dest, dest_names);
  return true;
}

}  // namespace internal

/**
 * @brief Append fast-access (and optional mag) copies to an open Dataset.
 *
 * New arrays are siblings of the source volume:
 *   - `fast_{dim}`: same dims / shape / coords as the source (Python FAP).
 *   - `fast_mag_{n}`: XY box-mean, Z kept; new dim names `{dim}_mag{n}`.
 *
 * Root `defaultVariableName` is left pointing at the original volume (or set
 * if missing). Callers that serve xarray / WEBKNOSSOS should treat `fast_*`
 * as layout variants, not extra volumes.
 *
 * @param dataset Open dataset. New variables are added in-memory and
 *        committed to durable metadata.
 * @param dataset_path Path used to create the new child arrays.
 * @param config Access-pattern and optional magnification config.
 */
inline Result<void> OptimizeAccessPatterns(
    Dataset& dataset, const std::string& dataset_path,
    const OptimizedAccessPatternConfig& config) {
  if (config.optimize_dimensions.empty() &&
      config.magnification_factors.empty()) {
    return absl::InvalidArgumentError(
        "Provide optimize_dimensions and/or magnification_factors.");
  }
  if (config.compressor_name != "blosc") {
    return absl::InvalidArgumentError(
        "Only blosc compressor is supported (lossless). ZFP is not available "
        "in this TensorStore build.");
  }
  if (config.compressor_level < 0 || config.compressor_level > 9) {
    return absl::InvalidArgumentError("Compressor level must be between 0 and 9");
  }
  if (config.max_threads < 0) {
    return absl::InvalidArgumentError("max_threads must be >= 0");
  }

  MDIO_ASSIGN_OR_RETURN(
      const std::string source_name,
      internal::ResolveSourceVariable(dataset, config.source_variable));
  MDIO_ASSIGN_OR_RETURN(auto source, dataset.variables.at(source_name));
  if (internal::LooksStructured(source)) {
    return absl::InvalidArgumentError(
        "Access-pattern generation requires a scalar volume, not a structured "
        "dtype.");
  }
  MDIO_ASSIGN_OR_RETURN(const auto version,
                        internal::DetectVariableVersion(source));

  const auto& meta = dataset.getMetadata();
  if (!meta.contains("attributes") ||
      !meta["attributes"].contains("defaultVariableName")) {
    dataset.MergeAttributes({{"defaultVariableName", source_name}});
  }

  int written_count = 0;
  for (const auto& [dim_name, chunks] : config.optimize_dimensions) {
    auto written = internal::WriteFastAccessVariable(
        dataset, source, dataset_path, version, dim_name, chunks, config);
    if (!written.ok()) {
      return written.status();
    }
    ++written_count;
  }
  std::vector<int> factors = config.magnification_factors;
  std::sort(factors.begin(), factors.end());
  factors.erase(std::unique(factors.begin(), factors.end()), factors.end());
  for (int factor : factors) {
    auto written = internal::WriteMagnification(
        dataset, source, dataset_path, version, factor, config);
    if (!written.ok()) {
      return written.status();
    }
    if (written.value()) {
      ++written_count;
    }
  }
  if (written_count == 0) {
    return absl::OkStatus();
  }

  auto rebuilt = internal::RebuildDomain(dataset);
  if (!rebuilt.ok()) {
    return rebuilt.status();
  }
  auto commit = dataset.CommitMetadata();
  return commit.status();
}

/**
 * @brief Opens a dataset, appends access-pattern copies, and commits.
 */
inline Result<void> RestoreSkippedVariableSpecs(
    const std::string& dataset_path, const nlohmann::json& metadata,
    const std::vector<nlohmann::json>& skipped,
    tensorstore::Context context) {
  if (skipped.empty()) {
    return absl::OkStatus();
  }
  MDIO_ASSIGN_OR_RETURN(
      auto current,
      mdio::internal::from_zmetadata(dataset_path, context).result());
  auto [cur_meta, cur_vars] = current;
  std::unordered_set<std::string> have;
  have.reserve(cur_vars.size());
  for (const auto& spec : cur_vars) {
    have.insert(internal::VariableNameFromSpec(spec));
  }
  auto merged = cur_vars;
  for (const auto& spec : skipped) {
    const auto name = internal::VariableNameFromSpec(spec);
    if (!name.empty() && !have.count(name)) {
      merged.push_back(spec);
    }
  }
  if (merged.size() == cur_vars.size()) {
    return absl::OkStatus();
  }
  const nlohmann::json& write_meta = cur_meta.empty() ? metadata : cur_meta;
  return mdio::internal::write_zmetadata(write_meta, merged, context).result();
}

inline Result<void> OptimizeAccessPatterns(
    const std::string& dataset_path,
    const OptimizedAccessPatternConfig& config,
    tensorstore::Context context = tensorstore::Context::Default()) {
  MDIO_ASSIGN_OR_RETURN(
      auto params,
      mdio::internal::from_zmetadata(dataset_path, context).result());
  auto [metadata, json_vars] = params;
  std::vector<nlohmann::json> openable;
  std::vector<nlohmann::json> skipped;
  openable.reserve(json_vars.size());
  for (const auto& spec : json_vars) {
    if (internal::ShouldSkipUnopenableLayout(internal::VariableNameFromSpec(spec))) {
      skipped.push_back(spec);
    } else {
      openable.push_back(spec);
    }
  }
  if (openable.empty()) {
    return absl::FailedPreconditionError(
        "No openable MDIO variables (all entries look like FAP/zfpy layouts).");
  }
  MDIO_ASSIGN_OR_RETURN(
      auto dataset,
      Dataset::Open(metadata, openable, constants::kOpen, context).result());
  auto optimized = OptimizeAccessPatterns(dataset, dataset_path, config);
  if (!optimized.status().ok()) {
    return optimized.status();
  }
  return RestoreSkippedVariableSpecs(dataset_path, metadata, skipped, context);
}

}  // namespace optimize
}  // namespace mdio

#endif  // MDIO_OPTIMIZE_ACCESS_PATTERN_H_
