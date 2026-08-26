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

#ifndef MDIO_LABELS_LABELS_H_
#define MDIO_LABELS_LABELS_H_

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/escaping.h"
#include "mdio/dataset.h"
#include "mdio/dataset_factory.h"
#include "mdio/optimize/access_pattern.h"
#include "mdio/variable.h"

// clang-format off
#include <nlohmann/json.hpp>  // NOLINT
// clang-format on

namespace mdio {
namespace labels {

inline constexpr const char* kLabelsVariable = "labels";
inline constexpr const char* kLabelDim = "label";
inline constexpr uint16_t kMaxSegmentId = 65535;

struct LabelRecord {
  std::string name;
  uint16_t identifier = 0;
  std::string group;
  int64_t creation_time = 0;
  nlohmann::json extra = nlohmann::json::object();
};

struct AnnotationBucket {
  int64_t x = 0;
  int64_t y = 0;
  int64_t z = 0;
  std::unordered_map<std::string, int64_t> additional;
  std::vector<uint8_t> bytes;
};

struct AnnotationDump {
  int bucket_width = 32;
  int bytes_per_voxel = 4;
  uint64_t largest_segment_id = 0;
  std::string wk_x_dim = "crossline";
  std::string wk_y_dim = "inline";
  std::string wk_z_dim = "sample";
  std::vector<LabelRecord> segments;
  std::vector<AnnotationBucket> buckets;
};

struct WriteLabelsConfig {
  std::string source_path;
  std::string dest_path;
  std::string source_variable;
  std::string variable_name = kLabelsVariable;
  int cache_mb = 1024;
  int threads = 0;
  int s3_concurrency = 64;
  std::function<void(int64_t done, int64_t total)> progress;
};

struct WriteLabelsResult {
  std::string dest_path;
  std::string variable_name;
  uint64_t label_size = 0;
  int64_t buckets_written = 0;
};

namespace internal {

inline uint64_t ReadVoxelId(const uint8_t* bytes, int nbytes) {
  uint64_t value = 0;
  const int n = std::min(nbytes, 8);
  for (int i = 0; i < n; ++i) {
    value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
  }
  return value;
}

inline Result<uint64_t> ResolveLabelSize(const AnnotationDump& dump) {
  uint64_t max_id = dump.largest_segment_id;
  for (const auto& rec : dump.segments) {
    max_id = std::max<uint64_t>(max_id, rec.identifier);
  }
  const int width = dump.bucket_width > 0 ? dump.bucket_width : 32;
  const int nbytes = dump.bytes_per_voxel > 0 ? dump.bytes_per_voxel : 4;
  const size_t expected = static_cast<size_t>(width) * width * width * nbytes;
  for (const auto& bucket : dump.buckets) {
    if (bucket.bytes.size() < expected) {
      continue;
    }
    for (size_t i = 0; i < expected; i += static_cast<size_t>(nbytes)) {
      const uint64_t id = ReadVoxelId(bucket.bytes.data() + i, nbytes);
      if (id > kMaxSegmentId) {
        return absl::InvalidArgumentError(
            "Segment id " + std::to_string(id) +
            " exceeds uint16 max (65535)");
      }
      max_id = std::max(max_id, id);
    }
  }
  if (max_id > kMaxSegmentId) {
    return absl::InvalidArgumentError(
        "largestSegmentId " + std::to_string(max_id) +
        " exceeds uint16 max (65535)");
  }
  return max_id + 1;
}

inline nlohmann::json RecordsToJson(const std::vector<LabelRecord>& records) {
  nlohmann::json out = nlohmann::json::array();
  for (const auto& rec : records) {
    nlohmann::json item = rec.extra.is_object() ? rec.extra
                                                : nlohmann::json::object();
    item["LabelName"] = rec.name.empty()
                            ? ("Segment " + std::to_string(rec.identifier))
                            : rec.name;
    item["LabelIdentifier"] = rec.identifier;
    if (!rec.group.empty()) {
      item["LabelGroup"] = rec.group;
    }
    if (rec.creation_time != 0) {
      item["LabelCreationTime"] = rec.creation_time;
    }
    out.push_back(std::move(item));
  }
  return out;
}

inline Result<nlohmann::json> CompressorFromVariable(const Variable<>& var) {
  MDIO_ASSIGN_OR_RETURN(auto spec, var.spec());
  MDIO_ASSIGN_OR_RETURN(auto json, spec.ToJson(mdio::IncludeDefaults{}));
  if (json.contains("metadata") && json["metadata"].contains("codecs") &&
      json["metadata"]["codecs"].is_array()) {
    for (const auto& codec : json["metadata"]["codecs"]) {
      if (codec.is_object() && codec.value("name", "") == "blosc") {
        const auto cfg = codec.value("configuration", nlohmann::json::object());
        return nlohmann::json{{"name", "blosc"},
                              {"algorithm", cfg.value("cname", "zstd")},
                              {"level", cfg.value("clevel", 5)}};
      }
    }
  }
  if (json.contains("compressor") && json["compressor"].is_object()) {
    const auto& c = json["compressor"];
    return nlohmann::json{
        {"name", "blosc"},
        {"algorithm", c.value("cname", c.value("algorithm", "zstd"))},
        {"level", c.value("clevel", c.value("level", 5))}};
  }
  return nlohmann::json{
      {"name", "blosc"}, {"algorithm", "zstd"}, {"level", 5}};
}

inline Result<std::vector<int64_t>> ChunkShape(const Variable<>& var) {
  MDIO_ASSIGN_OR_RETURN(auto chunks, var.get_chunk_shape());
  return std::vector<int64_t>(chunks.begin(), chunks.end());
}

inline Result<void> RequireV3(const Variable<>& var) {
  MDIO_ASSIGN_OR_RETURN(const auto version,
                        optimize::internal::DetectVariableVersion(var));
  if (version != zarr::ZarrVersion::kV3) {
    return absl::FailedPreconditionError(
        "MDIO labels export requires Zarr V3. Open reported a different "
        "version.");
  }
  return absl::OkStatus();
}

inline Result<nlohmann::json> CoordVariableSchema(const Variable<>& var) {
  const auto names = optimize::internal::DimensionNames(var);
  const auto sizes = optimize::internal::DimensionSizes(var);
  if (names.size() != sizes.size() || names.empty()) {
    return absl::InvalidArgumentError(
        "Coordinate variable '" + var.get_variable_name() +
        "' is missing dimension labels.");
  }
  MDIO_ASSIGN_OR_RETURN(const std::string dtype,
                        optimize::internal::DataTypeToMdioName(var.dtype()));
  std::vector<std::pair<std::string, uint64_t>> dims;
  dims.reserve(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    dims.emplace_back(names[i], static_cast<uint64_t>(sizes[i]));
  }
  std::vector<int64_t> chunks;
  auto chunk_res = ChunkShape(var);
  if (chunk_res.ok()) {
    chunks = *chunk_res;
  } else {
    for (auto size : sizes) {
      chunks.push_back(size);
    }
  }
  return optimize::internal::BuildVariableSchema(
      var.get_variable_name(), dtype, dims, chunks, {}, var.get_long_name(),
      nlohmann::json::object());
}

inline Result<void> CopyVariableData(Variable<> src, Variable<> dst) {
  auto read = src.Read().result();
  if (!read.status().ok()) {
    return read.status();
  }
  auto write = dst.Write(read.value());
  if (!write.status().ok()) {
    return write.status();
  }
  return absl::OkStatus();
}

inline Result<Dataset> CreateSidecar(
    const Dataset& source, const std::string& dest_path,
    const std::string& skip_volume, tensorstore::Context context) {
  nlohmann::json schema;
  schema["metadata"] = source.getMetadata();
  if (schema["metadata"].contains("attributes") &&
      schema["metadata"]["attributes"].is_object()) {
    schema["metadata"]["attributes"].erase("labels");
  }
  if (schema["metadata"].contains("name") &&
      schema["metadata"]["name"].is_string()) {
    schema["metadata"]["name"] =
        schema["metadata"]["name"].get<std::string>() + "_labels";
  }
  schema["variables"] = nlohmann::json::array();

  for (const auto& key : source.variables.get_iterable_accessor()) {
    if (key == skip_volume || key == kLabelsVariable || key == kLabelDim ||
        optimize::internal::IsLayoutVariantName(key)) {
      continue;
    }
    MDIO_ASSIGN_OR_RETURN(auto var, source.variables.at(key));
    if (optimize::internal::LooksStructured(var)) {
      continue;
    }
    if (!optimize::internal::IsCoordinateLikeName(key) && var.rank() != 1) {
      continue;
    }
    MDIO_ASSIGN_OR_RETURN(auto var_schema, CoordVariableSchema(var));
    schema["variables"].push_back(var_schema);
  }
  if (schema["variables"].empty()) {
    return absl::FailedPreconditionError(
        "Sidecar create found no coordinate variables to copy.");
  }

  if (dest_path.find("://") == std::string::npos) {
    std::filesystem::remove_all(dest_path);
  }
  MDIO_ASSIGN_OR_RETURN(
      auto dest, Dataset::from_json(schema, dest_path, zarr::ZarrVersion::kV3,
                                    constants::kCreate, context)
                     .result());
  for (const auto& key : dest.variables.get_iterable_accessor()) {
    if (!source.variables.contains_key(key)) {
      continue;
    }
    MDIO_ASSIGN_OR_RETURN(auto src_var, source.variables.at(key));
    MDIO_ASSIGN_OR_RETURN(auto dst_var, dest.variables.at(key));
    MDIO_RETURN_IF_ERROR(CopyVariableData(src_var, dst_var));
  }
  return dest;
}

inline Result<void> WriteLabelCoordinate(Dataset& dataset,
                                         const std::string& dataset_path,
                                         zarr::ZarrVersion version,
                                         uint64_t label_size,
                                         tensorstore::Context context) {
  std::vector<std::pair<std::string, uint64_t>> dims = {
      {kLabelDim, label_size}};
  auto schema = optimize::internal::BuildVariableSchema(
      kLabelDim, "uint16", dims, {static_cast<int64_t>(label_size)}, {},
      kLabelDim, nlohmann::json::object());
  std::unordered_map<std::string, uint64_t> dim_map{{kLabelDim, label_size}};
  MDIO_ASSIGN_OR_RETURN(
      auto var, optimize::internal::CreateVariableFromSchema(
                    schema, dataset_path, version, dim_map, context));
  MDIO_ASSIGN_OR_RETURN(auto data, from_variable(var));
  auto* ptr = reinterpret_cast<uint16_t*>(
      data.get_data_accessor().byte_strided_origin_pointer().get());
  for (uint64_t i = 0; i < label_size; ++i) {
    ptr[i] = static_cast<uint16_t>(i);
  }
  auto write = var.Write(data);
  if (!write.status().ok()) {
    return write.status();
  }
  optimize::internal::RegisterVariable(dataset, var, {});
  return absl::OkStatus();
}

inline Result<Variable<>> CreateLabelsVariable(
    Dataset& dataset, const Variable<>& source, const std::string& dataset_path,
    zarr::ZarrVersion version, uint64_t label_size,
    const std::string& variable_name, tensorstore::Context context) {
  const auto dim_names = optimize::internal::DimensionNames(source);
  const auto dim_sizes = optimize::internal::DimensionSizes(source);
  MDIO_ASSIGN_OR_RETURN(auto src_chunks, ChunkShape(source));
  if (dim_names.size() != dim_sizes.size() ||
      src_chunks.size() != dim_names.size()) {
    return absl::InvalidArgumentError(
        "Source volume dimensions and chunks are inconsistent.");
  }

  std::vector<std::pair<std::string, uint64_t>> dims;
  std::unordered_map<std::string, uint64_t> dim_map;
  std::vector<int64_t> chunks;
  dims.emplace_back(kLabelDim, label_size);
  dim_map[kLabelDim] = label_size;
  chunks.push_back(static_cast<int64_t>(label_size));
  for (size_t i = 0; i < dim_names.size(); ++i) {
    dims.emplace_back(dim_names[i], static_cast<uint64_t>(dim_sizes[i]));
    dim_map[dim_names[i]] = static_cast<uint64_t>(dim_sizes[i]);
    chunks.push_back(src_chunks[i]);
  }

  std::vector<std::string> coordinates = {kLabelDim};
  if (dataset.coordinates.count(source.get_variable_name())) {
    for (const auto& c : dataset.coordinates.at(source.get_variable_name())) {
      coordinates.push_back(c);
    }
  } else {
    coordinates.insert(coordinates.end(), dim_names.begin(), dim_names.end());
  }

  MDIO_ASSIGN_OR_RETURN(auto compressor, CompressorFromVariable(source));
  auto schema = optimize::internal::BuildVariableSchema(
      variable_name, "uint16", dims, chunks, coordinates, "segmentation labels",
      compressor);
  schema["metadata"]["fill_value"] = 0;

  MDIO_ASSIGN_OR_RETURN(
      auto var, optimize::internal::CreateVariableFromSchema(
                    schema, dataset_path, version, dim_map, context));
  optimize::internal::RegisterVariable(dataset, var, coordinates);
  return var;
}

inline int DimIndex(const std::vector<std::string>& names,
                    const std::string& name) {
  for (size_t i = 0; i < names.size(); ++i) {
    if (optimize::internal::NormalizeName(names[i]) ==
        optimize::internal::NormalizeName(name)) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

inline Index AdditionalCoord(const AnnotationBucket& bucket,
                             const std::string& dim_name) {
  auto it = bucket.additional.find(dim_name);
  if (it == bucket.additional.end()) {
    it = bucket.additional.find(optimize::internal::NormalizeName(dim_name));
  }
  if (it != bucket.additional.end()) {
    return static_cast<Index>(it->second);
  }
  return 0;
}

inline bool BucketOverlapsChunk(const AnnotationBucket& bucket,
                                const std::vector<std::string>& dim_names,
                                const std::vector<Index>& origin,
                                const std::vector<Index>& extent, int x_i,
                                int y_i, int z_i, int width) {
  for (size_t i = 0; i < dim_names.size(); ++i) {
    const Index c0 = origin[i];
    const Index c1 = origin[i] + extent[i];
    Index b0 = 0;
    Index b1 = 1;
    const int ii = static_cast<int>(i);
    if (ii == x_i) {
      b0 = static_cast<Index>(bucket.x);
      b1 = b0 + width;
    } else if (ii == y_i) {
      b0 = static_cast<Index>(bucket.y);
      b1 = b0 + width;
    } else if (ii == z_i) {
      b0 = static_cast<Index>(bucket.z);
      b1 = b0 + width;
    } else {
      b0 = AdditionalCoord(bucket, dim_names[i]);
      b1 = b0 + 1;
    }
    if (b0 >= c1 || b1 <= c0) {
      return false;
    }
  }
  return true;
}

inline Result<void> ScatterBuckets(Variable<> labels, const Variable<>& source,
                                   const AnnotationDump& dump,
                                   uint64_t label_size,
                                   const WriteLabelsConfig& config) {
  const auto dim_names = optimize::internal::DimensionNames(source);
  const auto dim_sizes = optimize::internal::DimensionSizes(source);
  MDIO_ASSIGN_OR_RETURN(auto src_chunks, ChunkShape(source));
  const int x_i = DimIndex(dim_names, dump.wk_x_dim);
  const int y_i = DimIndex(dim_names, dump.wk_y_dim);
  const int z_i = DimIndex(dim_names, dump.wk_z_dim);
  if (x_i < 0 || y_i < 0 || z_i < 0) {
    return absl::InvalidArgumentError(
        "wkToMdio dims do not match the source volume (" + dump.wk_x_dim +
        ", " + dump.wk_y_dim + ", " + dump.wk_z_dim + ")");
  }
  if (src_chunks.size() != dim_names.size() ||
      dim_sizes.size() != dim_names.size()) {
    return absl::InvalidArgumentError(
        "Source volume dimensions and chunks are inconsistent.");
  }

  const int width = dump.bucket_width > 0 ? dump.bucket_width : 32;
  const int nbytes = dump.bytes_per_voxel > 0 ? dump.bytes_per_voxel : 4;
  const size_t bucket_bytes =
      static_cast<size_t>(width) * width * width * static_cast<size_t>(nbytes);
  for (const auto& bucket : dump.buckets) {
    if (bucket.bytes.size() < bucket_bytes) {
      return absl::InvalidArgumentError(
          "Bucket at (" + std::to_string(bucket.x) + "," +
          std::to_string(bucket.y) + "," + std::to_string(bucket.z) +
          ") is shorter than " + std::to_string(bucket_bytes) + " bytes");
    }
  }

  const size_t rank = dim_names.size();
  std::vector<Index> nchunks(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const Index chunk =
        src_chunks[i] > 0 ? src_chunks[i] : dim_sizes[i];
    if (chunk <= 0 || dim_sizes[i] <= 0) {
      return absl::InvalidArgumentError("Invalid source chunk/shape");
    }
    nchunks[i] = (dim_sizes[i] + chunk - 1) / chunk;
  }

  struct SpatialChunk {
    std::vector<Index> origin;
    std::vector<Index> extent;
  };
  std::vector<SpatialChunk> dirty;
  std::vector<Index> chunk_idx(rank, 0);
  while (true) {
    SpatialChunk chunk;
    chunk.origin.resize(rank);
    chunk.extent.resize(rank);
    for (size_t i = 0; i < rank; ++i) {
      const Index step =
          src_chunks[i] > 0 ? src_chunks[i] : dim_sizes[i];
      chunk.origin[i] = chunk_idx[i] * step;
      chunk.extent[i] =
          std::min(step, dim_sizes[i] - chunk.origin[i]);
    }
    bool hit = false;
    for (const auto& bucket : dump.buckets) {
      if (BucketOverlapsChunk(bucket, dim_names, chunk.origin, chunk.extent,
                              x_i, y_i, z_i, width)) {
        hit = true;
        break;
      }
    }
    if (hit) {
      dirty.push_back(std::move(chunk));
    }
    size_t inc = 0;
    for (; inc < rank; ++inc) {
      ++chunk_idx[inc];
      if (chunk_idx[inc] < nchunks[inc]) {
        break;
      }
      chunk_idx[inc] = 0;
    }
    if (inc == rank) {
      break;
    }
  }

  const int64_t total = static_cast<int64_t>(dirty.size());
  int64_t done = 0;
  if (config.progress) {
    config.progress(0, total);
  }

  for (const auto& chunk : dirty) {
    std::vector<RangeDescriptor<Index>> slices;
    slices.push_back({kLabelDim, 0, static_cast<Index>(label_size), 1});
    for (size_t i = 0; i < rank; ++i) {
      slices.push_back({dim_names[i], chunk.origin[i],
                        chunk.origin[i] + chunk.extent[i], 1});
    }
    MDIO_ASSIGN_OR_RETURN(auto tile, labels.slice(slices));
    MDIO_ASSIGN_OR_RETURN(auto buf, from_variable(tile));
    auto acc = buf.get_data_accessor();
    auto* data = reinterpret_cast<uint16_t*>(
        acc.byte_strided_origin_pointer().get());
    const size_t n = acc.num_elements();
    std::fill(data, data + n, static_cast<uint16_t>(0));
    bool any = false;

    auto write_voxel = [&](uint16_t id,
                           const std::vector<Index>& spatial) -> absl::Status {
      if (id == 0) {
        return absl::OkStatus();
      }
      if (static_cast<uint64_t>(id) >= label_size) {
        return absl::OutOfRangeError("label index " + std::to_string(id) +
                                     " >= " + std::to_string(label_size));
      }
      size_t idx = static_cast<size_t>(id);
      for (size_t i = 0; i < spatial.size(); ++i) {
        const Index local = spatial[i] - chunk.origin[i];
        if (local < 0 || local >= chunk.extent[i]) {
          return absl::OkStatus();
        }
        idx = idx * static_cast<size_t>(chunk.extent[i]) +
              static_cast<size_t>(local);
      }
      if (idx >= n) {
        return absl::OutOfRangeError("flattened label index out of range");
      }
      data[idx] = id;
      any = true;
      return absl::OkStatus();
    };

    for (const auto& bucket : dump.buckets) {
      if (!BucketOverlapsChunk(bucket, dim_names, chunk.origin, chunk.extent,
                               x_i, y_i, z_i, width)) {
        continue;
      }
      std::vector<Index> extra(rank, 0);
      for (size_t i = 0; i < rank; ++i) {
        extra[i] = AdditionalCoord(bucket, dim_names[i]);
      }
      for (int lz = 0; lz < width; ++lz) {
        for (int ly = 0; ly < width; ++ly) {
          for (int lx = 0; lx < width; ++lx) {
            const size_t vi =
                static_cast<size_t>(lx + ly * width + lz * width * width);
            const uint64_t raw =
                ReadVoxelId(bucket.bytes.data() + vi * nbytes, nbytes);
            if (raw == 0) {
              continue;
            }
            if (raw > kMaxSegmentId) {
              return absl::InvalidArgumentError(
                  "Segment id " + std::to_string(raw) +
                  " exceeds uint16 max (65535)");
            }
            std::vector<Index> spatial = extra;
            spatial[static_cast<size_t>(x_i)] =
                static_cast<Index>(bucket.x + lx);
            spatial[static_cast<size_t>(y_i)] =
                static_cast<Index>(bucket.y + ly);
            spatial[static_cast<size_t>(z_i)] =
                static_cast<Index>(bucket.z + lz);
            auto st = write_voxel(static_cast<uint16_t>(raw), spatial);
            if (!st.ok()) {
              return st;
            }
          }
        }
      }
    }
    if (any) {
      auto write = tile.Write(buf);
      if (!write.status().ok()) {
        return write.status();
      }
    }
    ++done;
    if (config.progress) {
      config.progress(done, total);
    }
  }
  return absl::OkStatus();
}

inline Result<void> ApplyLabelAttributes(Variable<>* labels,
                                         const std::vector<LabelRecord>& recs) {
  nlohmann::json attrs = labels->GetAttributes();
  if (!attrs.is_object()) {
    attrs = nlohmann::json::object();
  }
  if (!attrs.contains("attributes") || !attrs["attributes"].is_object()) {
    attrs["attributes"] = nlohmann::json::object();
  }
  attrs["attributes"]["labels"] = RecordsToJson(recs);
  return labels->UpdateAttributes(attrs);
}

}  // namespace internal

inline Result<AnnotationDump> LoadAnnotationDump(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return absl::NotFoundError("Annotation dump not found: " + path);
  }
  nlohmann::json json;
  in >> json;
  AnnotationDump dump;
  dump.bucket_width = json.value("bucketWidth", 32);
  dump.bytes_per_voxel = json.value("bytesPerVoxel", 4);
  dump.largest_segment_id = json.value("largestSegmentId", 0);
  if (json.contains("wkToMdio") && json["wkToMdio"].is_object()) {
    dump.wk_x_dim = json["wkToMdio"].value("x", dump.wk_x_dim);
    dump.wk_y_dim = json["wkToMdio"].value("y", dump.wk_y_dim);
    dump.wk_z_dim = json["wkToMdio"].value("z", dump.wk_z_dim);
  }
  if (json.contains("segments") && json["segments"].is_array()) {
    for (const auto& s : json["segments"]) {
      LabelRecord rec;
      rec.name = s.value("LabelName", "");
      rec.identifier = static_cast<uint16_t>(s.value("LabelIdentifier", 0));
      rec.group = s.value("LabelGroup", "");
      rec.creation_time = s.value("LabelCreationTime", 0);
      rec.extra = nlohmann::json::object();
      for (auto it = s.begin(); it != s.end(); ++it) {
        if (it.key() == "LabelName" || it.key() == "LabelIdentifier" ||
            it.key() == "LabelGroup" || it.key() == "LabelCreationTime") {
          continue;
        }
        rec.extra[it.key()] = it.value();
      }
      dump.segments.push_back(std::move(rec));
    }
  }
  const auto parent = std::filesystem::path(path).parent_path();
  if (json.contains("buckets") && json["buckets"].is_array()) {
    for (const auto& b : json["buckets"]) {
      AnnotationBucket bucket;
      bucket.x = b.value("x", 0);
      bucket.y = b.value("y", 0);
      bucket.z = b.value("z", 0);
      if (b.contains("additional") && b["additional"].is_object()) {
        for (auto it = b["additional"].begin(); it != b["additional"].end();
             ++it) {
          bucket.additional[it.key()] = it.value().get<int64_t>();
        }
      }
      if (b.contains("data") && b["data"].is_string()) {
        std::string decoded;
        if (!absl::Base64Unescape(b["data"].get<std::string>(), &decoded)) {
          return absl::InvalidArgumentError("Invalid base64 bucket data");
        }
        bucket.bytes.assign(decoded.begin(), decoded.end());
      } else if (b.contains("file") && b["file"].is_string()) {
        const auto file = parent / b["file"].get<std::string>();
        std::ifstream bin(file, std::ios::binary);
        if (!bin) {
          return absl::NotFoundError("Bucket file not found: " +
                                     file.string());
        }
        bucket.bytes.assign(std::istreambuf_iterator<char>(bin),
                            std::istreambuf_iterator<char>());
      } else {
        return absl::InvalidArgumentError(
            "Bucket is missing data or file field");
      }
      dump.buckets.push_back(std::move(bucket));
    }
  }
  return dump;
}

inline Result<WriteLabelsResult> WriteLabels(const WriteLabelsConfig& config,
                                             const AnnotationDump& dump,
                                             tensorstore::Context context) {
  if (config.source_path.empty()) {
    return absl::InvalidArgumentError("--source is required");
  }
  MDIO_ASSIGN_OR_RETURN(
      auto source_ds,
      Dataset::Open(config.source_path, constants::kOpen, context).result());
  MDIO_ASSIGN_OR_RETURN(
      const std::string source_name,
      optimize::internal::ResolveSourceVariable(source_ds,
                                                config.source_variable));
  MDIO_ASSIGN_OR_RETURN(auto source, source_ds.variables.at(source_name));
  MDIO_RETURN_IF_ERROR(internal::RequireV3(source));
  MDIO_ASSIGN_OR_RETURN(const uint64_t label_size,
                        internal::ResolveLabelSize(dump));
  MDIO_ASSIGN_OR_RETURN(const auto version,
                        optimize::internal::DetectVariableVersion(source));

  const std::string dest_path =
      config.dest_path.empty() ? config.source_path : config.dest_path;
  const bool inplace = dest_path == config.source_path;

  Dataset dest_ds = source_ds;
  if (!inplace) {
    MDIO_ASSIGN_OR_RETURN(
        dest_ds, internal::CreateSidecar(source_ds, dest_path, source_name,
                                         context));
  }

  MDIO_RETURN_IF_ERROR(internal::WriteLabelCoordinate(
      dest_ds, dest_path, version, label_size, context));
  MDIO_ASSIGN_OR_RETURN(
      auto labels_var,
      internal::CreateLabelsVariable(dest_ds, source, dest_path, version,
                                     label_size, config.variable_name,
                                     context));
  MDIO_RETURN_IF_ERROR(internal::ScatterBuckets(
      labels_var, source, dump, label_size, config));
  MDIO_ASSIGN_OR_RETURN(labels_var, dest_ds.variables.at(config.variable_name));
  MDIO_RETURN_IF_ERROR(
      internal::ApplyLabelAttributes(&labels_var, dump.segments));
  dest_ds.variables.add(config.variable_name, labels_var);
  MDIO_RETURN_IF_ERROR(optimize::internal::RebuildDomain(dest_ds));
  MDIO_RETURN_IF_ERROR(dest_ds.CommitMetadata().status());

  const auto meta = dest_ds.getMetadata();
  if (meta.contains("labels") ||
      (meta.contains("attributes") && meta["attributes"].is_object() &&
       meta["attributes"].contains("labels"))) {
    return absl::InternalError(
        "labels key leaked into dataset metadata; must stay on the Variable");
  }

  WriteLabelsResult result;
  result.dest_path = dest_path;
  result.variable_name = config.variable_name;
  result.label_size = label_size;
  result.buckets_written = static_cast<int64_t>(dump.buckets.size());
  return result;
}

inline Result<WriteLabelsResult> WriteLabels(const WriteLabelsConfig& config,
                                             const AnnotationDump& dump) {
  const int copy_limit =
      config.threads > 0
          ? config.threads
          : static_cast<int>(std::thread::hardware_concurrency());
  nlohmann::json ctx_json = {
      {"cache_pool",
       {{"total_bytes_limit",
         static_cast<uint64_t>(std::max(1, config.cache_mb)) * 1024ull *
             1024ull}}},
      {"data_copy_concurrency",
       {{"limit", copy_limit > 0 ? copy_limit : 32}}},
      {"s3_request_concurrency",
       {{"limit", std::max(1, config.s3_concurrency)}}}};
  auto spec = tensorstore::Context::Spec::FromJson(ctx_json);
  if (!spec.ok()) {
    spec = tensorstore::Context::Spec::FromJson(
        {{"cache_pool",
          {{"total_bytes_limit",
            static_cast<uint64_t>(std::max(1, config.cache_mb)) * 1024ull *
                1024ull}}}});
  }
  if (!spec.ok()) {
    return spec.status();
  }
  return WriteLabels(config, dump, tensorstore::Context(*spec));
}

}  // namespace labels
}  // namespace mdio

#endif  // MDIO_LABELS_LABELS_H_
