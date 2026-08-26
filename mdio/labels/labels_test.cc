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

#include "mdio/labels/labels.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/escaping.h"
#include "mdio/dataset.h"
#include "mdio/optimize/access_pattern.h"
#include "tensorstore/util/status_testutil.h"

// clang-format off
#include <nlohmann/json.hpp>  // NOLINT
// clang-format on

namespace {

::nlohmann::json GetSmallSchema() {
  return ::nlohmann::json::parse(R"(
{
  "metadata": {
    "name": "labels_test",
    "apiVersion": "1.0.0",
    "createdOn": "2023-12-12T15:02:06.413469-06:00",
    "attributes": {"foo": "bar"}
  },
  "variables": [
    {
      "name": "image",
      "dataType": "float32",
      "dimensions": [
        {"name": "inline", "size": 8},
        {"name": "crossline", "size": 16},
        {"name": "depth", "size": 12}
      ],
      "metadata": {
        "chunkGrid": {
          "name": "regular",
          "configuration": { "chunkShape": [8, 8, 12] }
        }
      },
      "coordinates": ["inline", "crossline", "depth"],
      "compressor": {"name": "blosc", "algorithm": "zstd"}
    },
    {
      "name": "inline",
      "dataType": "uint32",
      "dimensions": [{"name": "inline", "size": 8}]
    },
    {
      "name": "crossline",
      "dataType": "uint32",
      "dimensions": [{"name": "crossline", "size": 16}]
    },
    {
      "name": "depth",
      "dataType": "uint32",
      "dimensions": [{"name": "depth", "size": 12}]
    }
  ]
}
)");
}

mdio::Result<mdio::Dataset> MakeVolume(const std::string& path) {
  std::filesystem::remove_all(path);
  auto schema = GetSmallSchema();
  MDIO_ASSIGN_OR_RETURN(
      auto ds, mdio::Dataset::from_json(schema, path,
                                        mdio::zarr::ZarrVersion::kV3,
                                        mdio::constants::kCreateClean)
                   .result());
  return ds;
}

std::vector<uint8_t> EmptyBucket(int width = 32, int nbytes = 4) {
  return std::vector<uint8_t>(
      static_cast<size_t>(width) * width * width * nbytes, 0);
}

void SetVoxel(std::vector<uint8_t>* bucket, int x, int y, int z, uint32_t id,
              int width = 32, int nbytes = 4) {
  const size_t vi = static_cast<size_t>(x + y * width + z * width * width);
  for (int i = 0; i < nbytes; ++i) {
    (*bucket)[vi * nbytes + i] = static_cast<uint8_t>((id >> (8 * i)) & 0xff);
  }
}

TEST(Labels, InPlaceUint16FillAndAttrs) {
  const std::string path = "zarrs/labels_inplace.mdio";
  auto ds_res = MakeVolume(path);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();

  mdio::labels::AnnotationDump dump;
  dump.largest_segment_id = 5;
  dump.wk_x_dim = "crossline";
  dump.wk_y_dim = "inline";
  dump.wk_z_dim = "depth";
  dump.segments = {
      {"salt_body_1", 3, "salt", 1710000000000, {}},
      {"fault_1", 4, "fault", 0, {}},
      {"Segment 5", 5, "", 0, {{"Note", "rhs"}}},
  };
  auto bytes = EmptyBucket();
  SetVoxel(&bytes, 1, 2, 3, 3);
  SetVoxel(&bytes, 2, 2, 3, 4);
  SetVoxel(&bytes, 3, 2, 3, 5);
  SetVoxel(&bytes, 10, 2, 3, 3);  // second spatial chunk along crossline
  dump.buckets.push_back({0, 0, 0, {}, bytes});

  mdio::labels::WriteLabelsConfig config;
  config.source_path = path;
  config.source_variable = "image";
  auto result = mdio::labels::WriteLabels(config, dump);
  ASSERT_TRUE(result.status().ok()) << result.status();
  EXPECT_EQ(result->label_size, 6u);
  EXPECT_EQ(result->variable_name, "labels");

  auto reopened =
      mdio::Dataset::Open(path, mdio::constants::kOpen).result();
  ASSERT_TRUE(reopened.status().ok()) << reopened.status();
  auto ds = reopened.value();
  const auto meta = ds.getMetadata();
  EXPECT_FALSE(meta.contains("labels"));
  ASSERT_TRUE(meta.contains("attributes"));
  EXPECT_FALSE(meta["attributes"].contains("labels"));
  EXPECT_EQ(meta["attributes"]["foo"], "bar");

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto labels, ds.variables.at("labels"));
  const auto names = mdio::optimize::internal::DimensionNames(labels);
  ASSERT_GE(names.size(), 1u);
  EXPECT_EQ(names[0], "label");
  EXPECT_EQ(labels.dimensions().shape()[0], 6);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto chunks, labels.get_chunk_shape());
  ASSERT_GE(chunks.size(), 1u);
  EXPECT_EQ(chunks[0], 6);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, labels.spec());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_json,
                                   spec.ToJson(mdio::IncludeDefaults{}));
  ASSERT_TRUE(spec_json["metadata"].contains("fill_value"));
  EXPECT_EQ(spec_json["metadata"]["fill_value"], 0);

  auto attrs = labels.GetAttributes();
  ASSERT_TRUE(attrs.contains("attributes"));
  ASSERT_TRUE(attrs["attributes"].contains("labels"));
  ASSERT_EQ(attrs["attributes"]["labels"].size(), 3);
  EXPECT_EQ(attrs["attributes"]["labels"][0]["LabelName"], "salt_body_1");
  EXPECT_EQ(attrs["attributes"]["labels"][0]["LabelIdentifier"], 3);
  EXPECT_EQ(attrs["attributes"]["labels"][0]["LabelGroup"], "salt");
  EXPECT_FALSE(attrs["attributes"]["labels"][0].contains("LabelColor"));
  EXPECT_FALSE(attrs["attributes"]["labels"][0].contains("LabelGroupId"));
  EXPECT_FALSE(attrs["attributes"]["labels"][0].contains("LabelVisible"));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read, labels.Read().result());
  auto acc = read.get_data_accessor();
  auto* data = reinterpret_cast<uint16_t*>(
      acc.byte_strided_origin_pointer().get());
  // shape (6, 8, 16, 12) — index [label, inline, crossline, depth]
  auto at = [&](int lab, int inl, int xl, int dep) -> uint16_t {
    return data[((((lab * 8 + inl) * 16 + xl) * 12) + dep)];
  };
  EXPECT_EQ(at(3, 2, 1, 3), 3);
  EXPECT_EQ(at(4, 2, 2, 3), 4);
  EXPECT_EQ(at(5, 2, 3, 3), 5);
  EXPECT_EQ(at(3, 2, 10, 3), 3);
  EXPECT_EQ(at(3, 0, 0, 0), 0);
  EXPECT_EQ(at(0, 2, 1, 3), 0);

  EXPECT_TRUE(std::filesystem::exists(
      std::filesystem::path(path) / "labels" / "zarr.json"));
  EXPECT_FALSE(std::filesystem::exists(
      std::filesystem::path(path) / "labels" / ".zattrs"));
}

TEST(Labels, SidecarDoesNotTouchSource) {
  const std::string source = "zarrs/labels_sidecar_src.mdio";
  const std::string dest = "zarrs/labels_sidecar_dst.mdio";
  auto ds_res = MakeVolume(source);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();

  mdio::labels::AnnotationDump dump;
  dump.largest_segment_id = 1;
  dump.wk_x_dim = "crossline";
  dump.wk_y_dim = "inline";
  dump.wk_z_dim = "depth";
  dump.segments = {{"one", 1, "", 0, {}}};
  auto bytes = EmptyBucket();
  SetVoxel(&bytes, 0, 0, 0, 1);
  dump.buckets.push_back({0, 0, 0, {}, bytes});

  mdio::labels::WriteLabelsConfig config;
  config.source_path = source;
  config.dest_path = dest;
  config.source_variable = "image";
  auto result = mdio::labels::WriteLabels(config, dump);
  ASSERT_TRUE(result.status().ok()) << result.status();

  auto src = mdio::Dataset::Open(source, mdio::constants::kOpen).result();
  ASSERT_TRUE(src.status().ok()) << src.status();
  EXPECT_FALSE(src->variables.contains_key("labels"));

  auto dst = mdio::Dataset::Open(dest, mdio::constants::kOpen).result();
  ASSERT_TRUE(dst.status().ok()) << dst.status();
  EXPECT_TRUE(dst->variables.contains_key("labels"));
  EXPECT_TRUE(dst->variables.contains_key("inline"));
  EXPECT_FALSE(dst->variables.contains_key("image"));
  EXPECT_FALSE(dst->getMetadata().contains("labels"));
  if (dst->getMetadata().contains("attributes")) {
    EXPECT_FALSE(dst->getMetadata()["attributes"].contains("labels"));
  }
}

TEST(Labels, RejectsIdAboveUint16) {
  const std::string path = "zarrs/labels_overflow.mdio";
  auto ds_res = MakeVolume(path);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();

  mdio::labels::AnnotationDump dump;
  dump.largest_segment_id = 70000;
  dump.wk_x_dim = "crossline";
  dump.wk_y_dim = "inline";
  dump.wk_z_dim = "depth";
  mdio::labels::WriteLabelsConfig config;
  config.source_path = path;
  config.source_variable = "image";
  auto result = mdio::labels::WriteLabels(config, dump);
  EXPECT_FALSE(result.status().ok());
}

TEST(Labels, LoadDumpRoundTrip) {
  const std::string dir = "zarrs/labels_dump";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  auto bytes = EmptyBucket();
  SetVoxel(&bytes, 1, 0, 0, 7);
  const std::string encoded = absl::Base64Escape(
      std::string(bytes.begin(), bytes.end()));
  nlohmann::json json = {
      {"bucketWidth", 32},
      {"bytesPerVoxel", 4},
      {"largestSegmentId", 7},
      {"wkToMdio",
       {{"x", "crossline"}, {"y", "inline"}, {"z", "depth"}}},
      {"segments",
       {{{"LabelName", "salt_body_1"}, {"LabelIdentifier", 7}}}},
      {"buckets", {{{"x", 0}, {"y", 0}, {"z", 0}, {"data", encoded}}}},
  };
  const auto path = std::filesystem::path(dir) / "manifest.json";
  std::ofstream(path) << json.dump();
  auto dump = mdio::labels::LoadAnnotationDump(path.string());
  ASSERT_TRUE(dump.status().ok()) << dump.status();
  EXPECT_EQ(dump->largest_segment_id, 7u);
  ASSERT_EQ(dump->buckets.size(), 1u);
  EXPECT_EQ(dump->segments[0].name, "salt_body_1");
}

}  // namespace
