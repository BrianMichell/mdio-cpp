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

#include "mdio/optimize/access_pattern.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "mdio/dataset.h"
#include "mdio/zarr/zarr.h"
#include "tensorstore/util/status_testutil.h"

// clang-format off
#include <nlohmann/json.hpp>  // NOLINT
// clang-format on

namespace {

std::string ZarrVersionToString(mdio::zarr::ZarrVersion version) {
  return version == mdio::zarr::ZarrVersion::kV3 ? "V3" : "V2";
}

std::string GetBasePath(mdio::zarr::ZarrVersion version) {
  return version == mdio::zarr::ZarrVersion::kV3
             ? "zarrs/access_pattern_v3.mdio"
             : "zarrs/access_pattern.mdio";
}

::nlohmann::json GetSmallSchema(int inline_n = 8, int crossline_n = 16,
                                int depth_n = 12) {
  auto schema = ::nlohmann::json::parse(R"(
{
  "metadata": {
    "name": "fap_test",
    "apiVersion": "1.0.0",
    "createdOn": "2023-12-12T15:02:06.413469-06:00",
    "attributes": {}
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
  schema["variables"][0]["dimensions"][0]["size"] = inline_n;
  schema["variables"][0]["dimensions"][1]["size"] = crossline_n;
  schema["variables"][0]["dimensions"][2]["size"] = depth_n;
  schema["variables"][0]["metadata"]["chunkGrid"]["configuration"]
                    ["chunkShape"] = {std::min(inline_n, 8),
                                      std::min(crossline_n, 8),
                                      std::min(depth_n, 12)};
  schema["variables"][1]["dimensions"][0]["size"] = inline_n;
  schema["variables"][2]["dimensions"][0]["size"] = crossline_n;
  schema["variables"][3]["dimensions"][0]["size"] = depth_n;
  return schema;
}

mdio::Result<std::vector<int64_t>> GetChunks(const mdio::Variable<>& var) {
  MDIO_ASSIGN_OR_RETURN(auto spec, var.spec());
  MDIO_ASSIGN_OR_RETURN(auto json, spec.ToJson(mdio::IncludeDefaults{}));
  if (json["metadata"].contains("chunks")) {
    return json["metadata"]["chunks"].get<std::vector<int64_t>>();
  }
  if (json["metadata"].contains("chunk_grid")) {
    return json["metadata"]["chunk_grid"]["configuration"]["chunk_shape"]
        .get<std::vector<int64_t>>();
  }
  return absl::NotFoundError("Variable spec has no chunk metadata.");
}

float ImageValue(int inline_i, int crossline_i, int depth_i) {
  return static_cast<float>(inline_i * 1000 + crossline_i * 10 + depth_i);
}

float MeanImage(int inline0, int inline1, int crossline0, int crossline1,
                int depth_i) {
  double sum = 0.0;
  int count = 0;
  for (int i = inline0; i < inline1; ++i) {
    for (int j = crossline0; j < crossline1; ++j) {
      sum += ImageValue(i, j, depth_i);
      ++count;
    }
  }
  return static_cast<float>(sum / count);
}

mdio::Result<mdio::Dataset> MakePopulated(const std::string& path,
                                          mdio::zarr::ZarrVersion version,
                                          int inline_n = 8, int crossline_n = 16,
                                          int depth_n = 12) {
  std::filesystem::remove_all(path);
  auto schema = GetSmallSchema(inline_n, crossline_n, depth_n);
  MDIO_ASSIGN_OR_RETURN(
      auto ds, mdio::Dataset::from_json(schema, path, version,
                                        mdio::constants::kCreateClean)
                   .result());

  MDIO_ASSIGN_OR_RETURN(auto image,
                        ds.variables.get<mdio::dtypes::float32_t>("image"));
  MDIO_ASSIGN_OR_RETURN(auto inline_var,
                        ds.variables.get<mdio::dtypes::uint32_t>("inline"));
  MDIO_ASSIGN_OR_RETURN(auto crossline_var,
                        ds.variables.get<mdio::dtypes::uint32_t>("crossline"));
  MDIO_ASSIGN_OR_RETURN(auto depth_var,
                        ds.variables.get<mdio::dtypes::uint32_t>("depth"));

  MDIO_ASSIGN_OR_RETURN(auto image_data,
                        mdio::from_variable<mdio::dtypes::float32_t>(image));
  MDIO_ASSIGN_OR_RETURN(
      auto inline_data, mdio::from_variable<mdio::dtypes::uint32_t>(inline_var));
  MDIO_ASSIGN_OR_RETURN(
      auto crossline_data,
      mdio::from_variable<mdio::dtypes::uint32_t>(crossline_var));
  MDIO_ASSIGN_OR_RETURN(auto depth_data,
                        mdio::from_variable<mdio::dtypes::uint32_t>(depth_var));

  auto image_acc = image_data.get_data_accessor();
  auto inline_acc = inline_data.get_data_accessor();
  auto crossline_acc = crossline_data.get_data_accessor();
  auto depth_acc = depth_data.get_data_accessor();

  for (int i = 0; i < inline_n; ++i) {
    inline_acc({i}) = static_cast<uint32_t>(i * 10);
    for (int j = 0; j < crossline_n; ++j) {
      if (i == 0) {
        crossline_acc({j}) = static_cast<uint32_t>(j + 100);
      }
      for (int k = 0; k < depth_n; ++k) {
        if (i == 0 && j == 0) {
          depth_acc({k}) = static_cast<uint32_t>(k * 4);
        }
        image_acc({i, j, k}) = ImageValue(i, j, k);
      }
    }
  }

  MDIO_RETURN_IF_ERROR(image.Write(image_data).status());
  MDIO_RETURN_IF_ERROR(inline_var.Write(inline_data).status());
  MDIO_RETURN_IF_ERROR(crossline_var.Write(crossline_data).status());
  MDIO_RETURN_IF_ERROR(depth_var.Write(depth_data).status());
  return ds;
}

class AccessPatternVersionTest
    : public ::testing::TestWithParam<mdio::zarr::ZarrVersion> {
 protected:
  void SetUp() override {
    version_ = GetParam();
    base_path_ = GetBasePath(version_);
    std::filesystem::remove_all(base_path_);
  }
  void TearDown() override { std::filesystem::remove_all(base_path_); }

  mdio::zarr::ZarrVersion version_;
  std::string base_path_;
};

TEST_P(AccessPatternVersionTest, FastAccessSameGrid) {
  auto ds_res = MakePopulated(base_path_, version_);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();
  auto ds = ds_res.value();

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.optimize_dimensions = {
      {"inline", {2, 16, 12}},
      {"depth", {8, 16, 2}},
  };
  auto opt = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_TRUE(opt.status().ok()) << opt.status();

  EXPECT_TRUE(ds.variables.contains_key("fast_inline"));
  EXPECT_TRUE(ds.variables.contains_key("fast_depth"));
  EXPECT_FALSE(ds.variables.contains_key("image_inline"));
  EXPECT_EQ(ds.getMetadata()["attributes"]["defaultVariableName"], "image");

  auto reopen =
      mdio::Dataset::Open(base_path_, mdio::constants::kOpen).result();
  ASSERT_TRUE(reopen.status().ok()) << reopen.status();
  auto reopened = reopen.value();
  EXPECT_TRUE(reopened.variables.contains_key("image"));
  EXPECT_TRUE(reopened.variables.contains_key("fast_inline"));
  EXPECT_TRUE(reopened.variables.contains_key("fast_depth"));
  EXPECT_EQ(reopened.getMetadata()["attributes"]["defaultVariableName"],
            "image");

  auto fast = reopened.variables.get<mdio::dtypes::float32_t>("fast_inline");
  ASSERT_TRUE(fast.status().ok()) << fast.status();
  auto chunks = GetChunks(fast.value());
  ASSERT_TRUE(chunks.status().ok()) << chunks.status();
  EXPECT_EQ(chunks.value(), (std::vector<int64_t>{2, 16, 12}));

  const auto labels = fast.value().dimensions().labels();
  ASSERT_EQ(labels.size(), 3);
  EXPECT_EQ(labels[0], "inline");
  EXPECT_EQ(labels[1], "crossline");
  EXPECT_EQ(labels[2], "depth");
  EXPECT_EQ(fast.value().dimensions().shape()[0], 8);
  EXPECT_EQ(fast.value().dimensions().shape()[1], 16);
  EXPECT_EQ(fast.value().dimensions().shape()[2], 12);

  auto data = fast.value().Read().result();
  ASSERT_TRUE(data.status().ok()) << data.status();
  auto acc = data.value().get_data_accessor();
  EXPECT_FLOAT_EQ(acc({0, 0, 0}), ImageValue(0, 0, 0));
  EXPECT_FLOAT_EQ(acc({7, 15, 11}), ImageValue(7, 15, 11));
  EXPECT_FLOAT_EQ(acc({3, 8, 5}), ImageValue(3, 8, 5));
}

TEST_P(AccessPatternVersionTest, MagnificationKeepsXarraySaneDims) {
  auto ds_res = MakePopulated(base_path_, version_);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();
  auto ds = ds_res.value();

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.magnification_factors = {2};
  auto opt = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_TRUE(opt.status().ok()) << opt.status();

  auto reopen =
      mdio::Dataset::Open(base_path_, mdio::constants::kOpen).result();
  ASSERT_TRUE(reopen.status().ok()) << reopen.status();
  auto reopened = reopen.value();

  EXPECT_TRUE(reopened.variables.contains_key("image"));
  EXPECT_TRUE(reopened.variables.contains_key("fast_mag_2"));
  EXPECT_TRUE(reopened.variables.contains_key("inline_mag2"));
  EXPECT_TRUE(reopened.variables.contains_key("crossline_mag2"));
  EXPECT_FALSE(reopened.variables.contains_key("image_2"));
  EXPECT_EQ(reopened.getMetadata()["attributes"]["defaultVariableName"],
            "image");

  auto mag = reopened.variables.get<mdio::dtypes::float32_t>("fast_mag_2");
  ASSERT_TRUE(mag.status().ok()) << mag.status();
  const auto labels = mag.value().dimensions().labels();
  ASSERT_EQ(labels.size(), 3);
  EXPECT_EQ(labels[0], "inline_mag2");
  EXPECT_EQ(labels[1], "crossline_mag2");
  EXPECT_EQ(labels[2], "depth");
  EXPECT_EQ(mag.value().dimensions().shape()[0], 4);
  EXPECT_EQ(mag.value().dimensions().shape()[1], 8);
  EXPECT_EQ(mag.value().dimensions().shape()[2], 12);

  auto data = mag.value().Read().result();
  ASSERT_TRUE(data.status().ok()) << data.status();
  auto acc = data.value().get_data_accessor();
  EXPECT_FLOAT_EQ(acc({0, 0, 0}), MeanImage(0, 2, 0, 2, 0));
  EXPECT_FLOAT_EQ(acc({1, 2, 5}), MeanImage(2, 4, 4, 6, 5));
  EXPECT_FLOAT_EQ(acc({3, 7, 11}), MeanImage(6, 8, 14, 16, 11));

  auto inline_mag =
      reopened.variables.get<mdio::dtypes::uint32_t>("inline_mag2");
  ASSERT_TRUE(inline_mag.status().ok()) << inline_mag.status();
  auto inline_data = inline_mag.value().Read().result();
  ASSERT_TRUE(inline_data.status().ok()) << inline_data.status();
  auto inline_acc = inline_data.value().get_data_accessor();
  EXPECT_EQ(inline_acc({0}), 0u);
  EXPECT_EQ(inline_acc({1}), 20u);
  EXPECT_EQ(inline_acc({3}), 60u);
}

TEST_P(AccessPatternVersionTest, FastAccessAndMagTogether) {
  auto ds_res = MakePopulated(base_path_, version_);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();
  auto ds = ds_res.value();

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.optimize_dimensions = {{"inline", {2, 16, 12}}};
  config.magnification_factors = {2};
  auto opt = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_TRUE(opt.status().ok()) << opt.status();

  auto reopen =
      mdio::Dataset::Open(base_path_, mdio::constants::kOpen).result();
  ASSERT_TRUE(reopen.status().ok()) << reopen.status();
  EXPECT_TRUE(reopen->variables.contains_key("fast_inline"));
  EXPECT_TRUE(reopen->variables.contains_key("fast_mag_2"));
  EXPECT_TRUE(reopen->variables.contains_key("image"));
}

TEST_P(AccessPatternVersionTest, MagnificationOddSourceSize) {
  auto ds_res = MakePopulated(base_path_, version_, 9, 17, 12);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();
  auto ds = ds_res.value();

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.magnification_factors = {2};
  auto opt = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_TRUE(opt.status().ok()) << opt.status();

  auto reopen =
      mdio::Dataset::Open(base_path_, mdio::constants::kOpen).result();
  ASSERT_TRUE(reopen.status().ok()) << reopen.status();
  auto mag = reopen->variables.get<mdio::dtypes::float32_t>("fast_mag_2");
  ASSERT_TRUE(mag.status().ok()) << mag.status();
  EXPECT_EQ(mag.value().dimensions().shape()[0], 5);
  EXPECT_EQ(mag.value().dimensions().shape()[1], 9);
  EXPECT_EQ(mag.value().dimensions().shape()[2], 12);

  auto data = mag.value().Read().result();
  ASSERT_TRUE(data.status().ok()) << data.status();
  auto acc = data.value().get_data_accessor();
  EXPECT_FLOAT_EQ(acc({0, 0, 0}), MeanImage(0, 2, 0, 2, 0));
  EXPECT_FLOAT_EQ(acc({4, 8, 11}), MeanImage(8, 9, 16, 17, 11));
}

TEST_P(AccessPatternVersionTest, RecreatesStaleMagCoord) {
  auto ds_res = MakePopulated(base_path_, version_, 9, 17, 12);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();
  auto ds = ds_res.value();

  std::unordered_map<std::string, uint64_t> dim_map{{"inline_mag2", 4}};
  auto leftover_schema = mdio::optimize::internal::BuildVariableSchema(
      "inline_mag2", "uint32", {{"inline_mag2", 4}}, {4}, {}, "inline_mag2",
      nlohmann::json());
  auto leftover = mdio::optimize::internal::CreateVariableFromSchema(
      leftover_schema, base_path_, version_, dim_map, ds.getContext());
  ASSERT_TRUE(leftover.status().ok()) << leftover.status();
  mdio::optimize::internal::RegisterVariable(ds, leftover.value(), {});
  EXPECT_EQ(leftover.value().dimensions().shape()[0], 4);

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.magnification_factors = {2};
  auto opt = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_TRUE(opt.status().ok()) << opt.status();

  auto reopen =
      mdio::Dataset::Open(base_path_, mdio::constants::kOpen).result();
  ASSERT_TRUE(reopen.status().ok()) << reopen.status();
  auto coord = reopen->variables.get<mdio::dtypes::uint32_t>("inline_mag2");
  ASSERT_TRUE(coord.status().ok()) << coord.status();
  EXPECT_EQ(coord.value().dimensions().shape()[0], 5);
  auto mag = reopen->variables.get<mdio::dtypes::float32_t>("fast_mag_2");
  ASSERT_TRUE(mag.status().ok()) << mag.status();
  EXPECT_EQ(mag.value().dimensions().shape()[0], 5);
}

TEST_P(AccessPatternVersionTest, ParallelTwoThreadsSameResult) {
  auto ds_res = MakePopulated(base_path_, version_);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();
  auto ds = ds_res.value();

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.magnification_factors = {2};
  config.max_threads = 2;
  config.max_memory_bytes = 4ull * 1024ull * 1024ull;
  config.processing_chunks = {{"inline", 2}, {"crossline", 2}, {"depth", 3}};
  auto opt = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_TRUE(opt.status().ok()) << opt.status();

  auto reopen =
      mdio::Dataset::Open(base_path_, mdio::constants::kOpen).result();
  ASSERT_TRUE(reopen.status().ok()) << reopen.status();
  auto mag = reopen->variables.get<mdio::dtypes::float32_t>("fast_mag_2");
  ASSERT_TRUE(mag.status().ok()) << mag.status();
  auto data = mag.value().Read().result();
  ASSERT_TRUE(data.status().ok()) << data.status();
  auto acc = data.value().get_data_accessor();
  EXPECT_FLOAT_EQ(acc({0, 0, 0}), MeanImage(0, 2, 0, 2, 0));
  EXPECT_FLOAT_EQ(acc({1, 2, 5}), MeanImage(2, 4, 4, 6, 5));
  EXPECT_FLOAT_EQ(acc({3, 7, 11}), MeanImage(6, 8, 14, 16, 11));
}

TEST_P(AccessPatternVersionTest, SkipsExistingMagnification) {
  auto ds_res = MakePopulated(base_path_, version_);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();
  auto ds = ds_res.value();

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.magnification_factors = {2};
  auto first = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_TRUE(first.status().ok()) << first.status();
  auto second = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_TRUE(second.status().ok()) << second.status();
  EXPECT_TRUE(ds.variables.contains_key("fast_mag_2"));
}

TEST_P(AccessPatternVersionTest, PyramidMag4MatchesFullResMean) {
  auto ds_res = MakePopulated(base_path_, version_);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();
  auto ds = ds_res.value();

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.magnification_factors = {2, 4};
  auto opt = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_TRUE(opt.status().ok()) << opt.status();

  auto reopen =
      mdio::Dataset::Open(base_path_, mdio::constants::kOpen).result();
  ASSERT_TRUE(reopen.status().ok()) << reopen.status();
  EXPECT_TRUE(reopen->variables.contains_key("fast_mag_2"));
  EXPECT_TRUE(reopen->variables.contains_key("fast_mag_4"));
  auto mag = reopen->variables.get<mdio::dtypes::float32_t>("fast_mag_4");
  ASSERT_TRUE(mag.status().ok()) << mag.status();
  EXPECT_EQ(mag.value().dimensions().shape()[0], 2);
  EXPECT_EQ(mag.value().dimensions().shape()[1], 4);
  EXPECT_EQ(mag.value().dimensions().shape()[2], 12);
  const auto labels = mag.value().dimensions().labels();
  ASSERT_EQ(labels.size(), 3);
  EXPECT_EQ(labels[0], "inline_mag4");
  EXPECT_EQ(labels[1], "crossline_mag4");
  EXPECT_EQ(labels[2], "depth");

  auto data = mag.value().Read().result();
  ASSERT_TRUE(data.status().ok()) << data.status();
  auto acc = data.value().get_data_accessor();
  EXPECT_FLOAT_EQ(acc({0, 0, 0}), MeanImage(0, 4, 0, 4, 0));
  EXPECT_FLOAT_EQ(acc({1, 2, 5}), MeanImage(4, 8, 8, 12, 5));
  EXPECT_FLOAT_EQ(acc({1, 3, 11}), MeanImage(4, 8, 12, 16, 11));
}

TEST_P(AccessPatternVersionTest, PyramidOddSourceSize) {
  auto ds_res = MakePopulated(base_path_, version_, 9, 17, 12);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();
  auto ds = ds_res.value();

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.magnification_factors = {2, 4};
  auto opt = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_TRUE(opt.status().ok()) << opt.status();

  auto reopen =
      mdio::Dataset::Open(base_path_, mdio::constants::kOpen).result();
  ASSERT_TRUE(reopen.status().ok()) << reopen.status();
  auto mag2 = reopen->variables.get<mdio::dtypes::float32_t>("fast_mag_2");
  ASSERT_TRUE(mag2.status().ok()) << mag2.status();
  EXPECT_EQ(mag2.value().dimensions().shape()[0], 5);
  EXPECT_EQ(mag2.value().dimensions().shape()[1], 9);
  EXPECT_EQ(mag2.value().dimensions().shape()[2], 12);

  auto mag4 = reopen->variables.get<mdio::dtypes::float32_t>("fast_mag_4");
  ASSERT_TRUE(mag4.status().ok()) << mag4.status();
  EXPECT_EQ(mag4.value().dimensions().shape()[0], 3);
  EXPECT_EQ(mag4.value().dimensions().shape()[1], 5);
  EXPECT_EQ(mag4.value().dimensions().shape()[2], 12);

  auto data = mag4.value().Read().result();
  ASSERT_TRUE(data.status().ok()) << data.status();
  auto acc = data.value().get_data_accessor();
  EXPECT_FLOAT_EQ(acc({0, 0, 0}), MeanImage(0, 4, 0, 4, 0));
  EXPECT_FLOAT_EQ(acc({2, 4, 11}), MeanImage(8, 9, 16, 17, 11));
}

TEST_P(AccessPatternVersionTest, Mag8OnlyFallsBackToFullRes) {
  auto ds_res = MakePopulated(base_path_, version_);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();
  auto ds = ds_res.value();

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.magnification_factors = {8};
  auto opt = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_TRUE(opt.status().ok()) << opt.status();

  auto reopen =
      mdio::Dataset::Open(base_path_, mdio::constants::kOpen).result();
  ASSERT_TRUE(reopen.status().ok()) << reopen.status();
  EXPECT_FALSE(reopen->variables.contains_key("fast_mag_2"));
  EXPECT_FALSE(reopen->variables.contains_key("fast_mag_4"));
  auto mag = reopen->variables.get<mdio::dtypes::float32_t>("fast_mag_8");
  ASSERT_TRUE(mag.status().ok()) << mag.status();
  EXPECT_EQ(mag.value().dimensions().shape()[0], 1);
  EXPECT_EQ(mag.value().dimensions().shape()[1], 2);
  EXPECT_EQ(mag.value().dimensions().shape()[2], 12);

  auto data = mag.value().Read().result();
  ASSERT_TRUE(data.status().ok()) << data.status();
  auto acc = data.value().get_data_accessor();
  EXPECT_FLOAT_EQ(acc({0, 0, 0}), MeanImage(0, 8, 0, 8, 0));
  EXPECT_FLOAT_EQ(acc({0, 1, 5}), MeanImage(0, 8, 8, 16, 5));
}

TEST_P(AccessPatternVersionTest, RejectsUnknownDimension) {
  auto ds_res = MakePopulated(base_path_, version_);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();
  auto ds = ds_res.value();

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.optimize_dimensions = {{"offset", {2, 16, 12}}};
  auto opt = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_FALSE(opt.status().ok());
  EXPECT_THAT(opt.status().message(), testing::HasSubstr("offset"));
}

TEST_P(AccessPatternVersionTest, RejectsExistingFastVariable) {
  auto ds_res = MakePopulated(base_path_, version_);
  ASSERT_TRUE(ds_res.status().ok()) << ds_res.status();
  auto ds = ds_res.value();

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.optimize_dimensions = {{"inline", {2, 16, 12}}};
  auto first = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_TRUE(first.status().ok()) << first.status();
  auto second = mdio::optimize::OptimizeAccessPatterns(ds, base_path_, config);
  ASSERT_FALSE(second.status().ok());
  EXPECT_THAT(second.status().message(), testing::HasSubstr("already exists"));
}

TEST(AccessPattern, RejectsEmptyConfig) {
  mdio::optimize::OptimizedAccessPatternConfig config;
  auto schema = GetSmallSchema();
  const std::string path = "zarrs/access_pattern_empty.mdio";
  std::filesystem::remove_all(path);
  auto ds = mdio::Dataset::from_json(schema, path, mdio::constants::kCreateClean)
                .result();
  ASSERT_TRUE(ds.status().ok()) << ds.status();
  auto opt =
      mdio::optimize::OptimizeAccessPatterns(ds.value(), path, config);
  ASSERT_FALSE(opt.status().ok());
  EXPECT_THAT(opt.status().message(),
              testing::HasSubstr("optimize_dimensions"));
  std::filesystem::remove_all(path);
}

TEST(AccessPattern, RejectsNonBloscCompressor) {
  mdio::optimize::OptimizedAccessPatternConfig config;
  config.optimize_dimensions = {{"inline", {2, 16, 12}}};
  config.compressor_name = "zfp";
  auto schema = GetSmallSchema();
  const std::string path = "zarrs/access_pattern_zfp.mdio";
  std::filesystem::remove_all(path);
  auto ds = mdio::Dataset::from_json(schema, path, mdio::constants::kCreateClean)
                .result();
  ASSERT_TRUE(ds.status().ok()) << ds.status();
  auto opt =
      mdio::optimize::OptimizeAccessPatterns(ds.value(), path, config);
  ASSERT_FALSE(opt.status().ok());
  EXPECT_THAT(opt.status().message(), testing::HasSubstr("blosc"));
  std::filesystem::remove_all(path);
}

INSTANTIATE_TEST_SUITE_P(
    ZarrVersions, AccessPatternVersionTest,
    ::testing::Values(mdio::zarr::ZarrVersion::kV2,
                      mdio::zarr::ZarrVersion::kV3),
    [](const ::testing::TestParamInfo<mdio::zarr::ZarrVersion>& info) {
      return ZarrVersionToString(info.param);
    });

}  // namespace
