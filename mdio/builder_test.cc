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

#include "mdio/builder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "mdio/dataset.h"
#include "mdio/dataset_validator.h"
#include "tensorstore/util/status_testutil.h"

// clang-format off
#include <nlohmann/json.hpp>  // NOLINT
// clang-format on

namespace {

using mdio::builder::CdpGatherDomain;
using mdio::builder::DatasetTemplate;
using mdio::builder::GetTemplate;
using mdio::builder::IsTemplateRegistered;
using mdio::builder::LengthUnit;
using mdio::builder::ListTemplates;
using mdio::builder::ParseCdpGatherDomain;
using mdio::builder::ParseSeismicDataDomain;
using mdio::builder::PostStack2DSpec;
using mdio::builder::PostStack3DSpec;
using mdio::builder::RegisterTemplate;
using mdio::builder::ScalarType;
using mdio::builder::SeismicDataDomain;
using mdio::builder::SingleNodeContRecvrGathersSpec;
using mdio::builder::StructuredType;
using mdio::builder::TemplateRegistry;
using mdio::builder::TemplateSpec;
using mdio::builder::TimeUnit;

const std::vector<std::string> kExpectedDefaultTemplateNames = {
    "PostStack2DTime",           "PostStack2DDepth",
    "PostStack3DTime",           "PostStack3DDepth",
    "CdpOffsetGathers2DTime",    "CdpOffsetGathers2DDepth",
    "CdpAngleGathers2DTime",     "CdpAngleGathers2DDepth",
    "CdpOffsetGathers3DTime",    "CdpOffsetGathers3DDepth",
    "CdpAngleGathers3DTime",     "CdpAngleGathers3DDepth",
    "CocaGathers3DTime",         "CocaGathers3DDepth",
    "ReceiverGathers3D",         "OffsetTiles3DTime",
    "OffsetTiles3DDepth",        "StreamerShotGathers2D",
    "StreamerShotGathers3D",     "StreamerFieldRecords3D",
    "ObnReceiverGathers3D",      "SingleNodeContRecvrGathers",
    "ShotReceiverLineGathers3D",
};

nlohmann::json StructuredHeaders() {
  return StructuredType{{
                            {"cdp_x", ScalarType::kInt32},
                            {"cdp_y", ScalarType::kInt32},
                            {"elevation", ScalarType::kFloat16},
                            {"some_scalar", ScalarType::kFloat16},
                        }}
      .ToJson();
}

const nlohmann::json* FindVariable(const nlohmann::json& dataset,
                                   const std::string& name) {
  for (const auto& var : dataset["variables"]) {
    if (var["name"] == name) {
      return &var;
    }
  }
  return nullptr;
}

std::vector<int64_t> DefaultSizesFor(const DatasetTemplate& templ) {
  std::vector<int64_t> sizes(templ.dimension_names().size(), 4);
  if (!sizes.empty()) {
    sizes.back() = 8;
  }
  return sizes;
}

TemplateSpec CustomTemplateSpec(std::string name) {
  TemplateSpec spec;
  spec.name = std::move(name);
  spec.data_domain = SeismicDataDomain::kTime;
  spec.dims = {{"cdp"}, {"time"}};
  spec.coords = {{"cdp_x", {"cdp"}}, {"cdp_y", {"cdp"}}};
  spec.chunks = {8, 8};
  spec.attributes = {{"surveyType", "2D"}, {"gatherType", "custom"}};
  return spec;
}

class RegistryTest : public ::testing::Test {
 protected:
  void TearDown() override { TemplateRegistry::ResetInstanceForTesting(); }
};

TEST_F(RegistryTest, DefaultCountIsTwentyThree) {
  const auto names = ListTemplates();
  EXPECT_EQ(names.size(), 23);
  EXPECT_EQ(kExpectedDefaultTemplateNames.size(), 23);
  for (const auto& expected : kExpectedDefaultTemplateNames) {
    EXPECT_TRUE(IsTemplateRegistered(expected)) << expected;
  }
}

TEST_F(RegistryTest, GetReturnsIndependentCopies) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto first, GetTemplate("PostStack3DTime"));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto second, GetTemplate("PostStack3DTime"));
  EXPECT_EQ(first.name(), second.name());
  TENSORSTORE_EXPECT_OK(first.set_full_chunk_shape({4, 4, 4}));
  EXPECT_EQ(second.stored_chunk_shape(), (std::vector<int64_t>{128, 128, 128}));
}

TEST_F(RegistryTest, RegisterAndUnregisterCustom) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto name, RegisterTemplate(CustomTemplateSpec("CustomTemplateTime")));
  EXPECT_EQ(name, "CustomTemplateTime");
  EXPECT_TRUE(IsTemplateRegistered("CustomTemplateTime"));

  auto again = RegisterTemplate(CustomTemplateSpec("CustomTemplateTime"));
  EXPECT_FALSE(again.ok());

  TENSORSTORE_EXPECT_OK(
      TemplateRegistry::GetInstance().Unregister("CustomTemplateTime"));
  EXPECT_FALSE(IsTemplateRegistered("CustomTemplateTime"));
}

TEST_F(RegistryTest, GetMissingFails) {
  auto missing = GetTemplate("DoesNotExist");
  EXPECT_FALSE(missing.ok());
}

TEST(ParseDomain, AcceptsMixedCase) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto time, ParseSeismicDataDomain("Time"));
  EXPECT_EQ(time, SeismicDataDomain::kTime);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto depth, ParseSeismicDataDomain("DePTh"));
  EXPECT_EQ(depth, SeismicDataDomain::kDepth);
  EXPECT_FALSE(ParseSeismicDataDomain("space").ok());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto offset, ParseCdpGatherDomain("OFFSET"));
  EXPECT_EQ(offset, CdpGatherDomain::kOffset);
  EXPECT_FALSE(ParseCdpGatherDomain("azimuth").ok());
}

TEST(PostStack, NamesAndAttributes) {
  DatasetTemplate t2d(PostStack2DSpec(SeismicDataDomain::kTime));
  EXPECT_EQ(t2d.name(), "PostStack2DTime");
  EXPECT_EQ(t2d.dimension_names(), (std::vector<std::string>{"cdp", "time"}));
  EXPECT_EQ(t2d.physical_coordinate_names(),
            (std::vector<std::string>{"cdp_x", "cdp_y"}));
  EXPECT_EQ(t2d.full_chunk_shape(), (std::vector<int64_t>{1024, 1024}));
  EXPECT_EQ(t2d.dataset_attributes(),
            (nlohmann::json{{"surveyType", "2D"}, {"gatherType", "stacked"}}));
  EXPECT_EQ(t2d.default_variable_name(), "amplitude");

  DatasetTemplate t3d(PostStack3DSpec(SeismicDataDomain::kDepth));
  EXPECT_EQ(t3d.name(), "PostStack3DDepth");
  EXPECT_EQ(t3d.dimension_names(),
            (std::vector<std::string>{"inline", "crossline", "depth"}));
  EXPECT_EQ(t3d.full_chunk_shape(), (std::vector<int64_t>{128, 128, 128}));
  EXPECT_EQ(t3d.dataset_attributes()["gatherType"], "stacked");
}

TEST(PostStack, ChunkShapeMinusOneExpandsAfterBuild) {
  DatasetTemplate templ(PostStack2DSpec(SeismicDataDomain::kTime));
  TENSORSTORE_ASSERT_OK(templ.set_full_chunk_shape({32, -1}));
  EXPECT_EQ(templ.full_chunk_shape(), (std::vector<int64_t>{32, -1}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto dataset,
                                   templ.BuildDataset("test", {100, 200}));
  EXPECT_EQ(templ.full_chunk_shape(), (std::vector<int64_t>{32, 200}));
  EXPECT_EQ(templ.stored_chunk_shape(), (std::vector<int64_t>{32, -1}));
  const auto* amplitude = FindVariable(dataset, "amplitude");
  ASSERT_NE(amplitude, nullptr);
  EXPECT_EQ(
      (*amplitude)["metadata"]["chunkGrid"]["configuration"]["chunkShape"],
      (nlohmann::json{32, 200}));
}

TEST(PostStack, ChunkShapeRejectsBadRankAndValues) {
  DatasetTemplate templ(PostStack2DSpec(SeismicDataDomain::kTime));
  EXPECT_FALSE(templ.set_full_chunk_shape({32, 32, 32}).ok());
  EXPECT_FALSE(templ.set_full_chunk_shape({32, 0}).ok());
  EXPECT_FALSE(templ.set_full_chunk_shape({32, -2}).ok());
  TENSORSTORE_EXPECT_OK(templ.set_full_chunk_shape({32, -1}));
}

TEST(SingleNode, TimeOnly) {
  EXPECT_FALSE(IsTemplateRegistered("SingleNodeContRecvrGathersDepth"));
  DatasetTemplate time(SingleNodeContRecvrGathersSpec());
  EXPECT_EQ(time.name(), "SingleNodeContRecvrGathers");
  EXPECT_EQ(time.data_domain(), SeismicDataDomain::kTime);
  EXPECT_EQ(time.dimension_names(),
            (std::vector<std::string>{"component", "epoch", "time"}));
  EXPECT_EQ(time.synthesize_missing_dims(),
            (std::vector<std::string>{"component"}));
  EXPECT_TRUE(time.GetUnitByKey("epoch").has_value());
}

struct TemplateCase {
  std::string name;
  std::vector<std::string> dims;
  std::vector<std::string> physical;
  std::vector<std::string> logical;
  std::vector<int64_t> chunks;
  std::string survey_key;
  std::string survey_value;
  std::string gather_type;
  std::vector<std::string> calculated;
};

class CanonicalTemplateTest : public ::testing::TestWithParam<TemplateCase> {};

TEST_P(CanonicalTemplateTest, ConfigurationMatchesPython) {
  const TemplateCase& test_case = GetParam();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto templ, GetTemplate(test_case.name));
  EXPECT_EQ(templ.name(), test_case.name);
  EXPECT_EQ(templ.dimension_names(), test_case.dims);
  EXPECT_EQ(templ.physical_coordinate_names(), test_case.physical);
  EXPECT_EQ(templ.logical_coordinate_names(), test_case.logical);
  EXPECT_EQ(templ.stored_chunk_shape(), test_case.chunks);
  EXPECT_EQ(templ.calculated_dimension_names(), test_case.calculated);
  EXPECT_EQ(templ.default_variable_name(), "amplitude");
  const nlohmann::json attrs = templ.dataset_attributes();
  EXPECT_EQ(attrs[test_case.survey_key], test_case.survey_value);
  EXPECT_EQ(attrs["gatherType"], test_case.gather_type);
}

TEST_P(CanonicalTemplateTest, BuildDatasetValidatesAgainstSchema) {
  const TemplateCase& test_case = GetParam();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto templ, GetTemplate(test_case.name));
  TENSORSTORE_ASSERT_OK(templ.AddUnits(
      {{"cdp_x", LengthUnit(std::string(mdio::units::kMeters))},
       {"cdp_y", LengthUnit(std::string(mdio::units::kMeters))},
       {"time", TimeUnit(std::string(mdio::units::kSeconds))},
       {"depth", LengthUnit(std::string(mdio::units::kMeters))}}));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto dataset, templ.BuildDataset(test_case.name, DefaultSizesFor(templ),
                                       StructuredHeaders()));
  EXPECT_EQ(dataset["metadata"]["name"], test_case.name);
  EXPECT_EQ(dataset["metadata"]["attributes"]["defaultVariableName"],
            "amplitude");
  EXPECT_EQ(dataset["metadata"]["attributes"]["gatherType"],
            test_case.gather_type);
  ASSERT_NE(FindVariable(dataset, "amplitude"), nullptr);
  ASSERT_NE(FindVariable(dataset, "trace_mask"), nullptr);
  ASSERT_NE(FindVariable(dataset, "headers"), nullptr);

  nlohmann::json mutable_dataset = dataset;
  TENSORSTORE_EXPECT_OK(validate_schema(mutable_dataset));

  // Calculated dims (shot_index) have no 1-d coordinate variable. C++
  // Dataset::from_json still requires that pairing, so skip full validate.
  if (test_case.calculated.empty()) {
    TENSORSTORE_EXPECT_OK(validate_dataset(mutable_dataset));
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllCanonical, CanonicalTemplateTest,
    ::testing::Values(
        TemplateCase{"PostStack2DTime",
                     {"cdp", "time"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {1024, 1024},
                     "surveyType",
                     "2D",
                     "stacked",
                     {}},
        TemplateCase{"PostStack2DDepth",
                     {"cdp", "depth"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {1024, 1024},
                     "surveyType",
                     "2D",
                     "stacked",
                     {}},
        TemplateCase{"PostStack3DTime",
                     {"inline", "crossline", "time"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {128, 128, 128},
                     "surveyType",
                     "3D",
                     "stacked",
                     {}},
        TemplateCase{"PostStack3DDepth",
                     {"inline", "crossline", "depth"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {128, 128, 128},
                     "surveyType",
                     "3D",
                     "stacked",
                     {}},
        TemplateCase{"CdpOffsetGathers2DTime",
                     {"cdp", "offset", "time"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {16, 64, 1024},
                     "surveyType",
                     "2D",
                     "cdp",
                     {}},
        TemplateCase{"CdpOffsetGathers2DDepth",
                     {"cdp", "offset", "depth"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {16, 64, 1024},
                     "surveyType",
                     "2D",
                     "cdp",
                     {}},
        TemplateCase{"CdpAngleGathers2DTime",
                     {"cdp", "angle", "time"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {16, 64, 1024},
                     "surveyType",
                     "2D",
                     "cdp",
                     {}},
        TemplateCase{"CdpAngleGathers2DDepth",
                     {"cdp", "angle", "depth"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {16, 64, 1024},
                     "surveyType",
                     "2D",
                     "cdp",
                     {}},
        TemplateCase{"CdpOffsetGathers3DTime",
                     {"inline", "crossline", "offset", "time"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {8, 8, 32, 512},
                     "surveyType",
                     "3D",
                     "cdp",
                     {}},
        TemplateCase{"CdpOffsetGathers3DDepth",
                     {"inline", "crossline", "offset", "depth"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {8, 8, 32, 512},
                     "surveyType",
                     "3D",
                     "cdp",
                     {}},
        TemplateCase{"CdpAngleGathers3DTime",
                     {"inline", "crossline", "angle", "time"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {8, 8, 32, 512},
                     "surveyType",
                     "3D",
                     "cdp",
                     {}},
        TemplateCase{"CdpAngleGathers3DDepth",
                     {"inline", "crossline", "angle", "depth"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {8, 8, 32, 512},
                     "surveyType",
                     "3D",
                     "cdp",
                     {}},
        TemplateCase{"CocaGathers3DTime",
                     {"inline", "crossline", "offset", "azimuth", "time"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {8, 8, 32, 1, 1024},
                     "surveyType",
                     "3D",
                     "common_offset_common_azimuth",
                     {}},
        TemplateCase{"CocaGathers3DDepth",
                     {"inline", "crossline", "offset", "azimuth", "depth"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {8, 8, 32, 1, 1024},
                     "surveyType",
                     "3D",
                     "common_offset_common_azimuth",
                     {}},
        TemplateCase{
            "ReceiverGathers3D",
            {"receiver", "shot_line", "shot_point", "time"},
            {"receiver_x", "receiver_y", "source_coord_x", "source_coord_y"},
            {},
            {1, 1, 512, 4096},
            "surveyType",
            "3D",
            "receiver_gathers",
            {}},
        TemplateCase{"OffsetTiles3DTime",
                     {"inline", "crossline", "inline_offset_tile",
                      "crossline_offset_tile", "time"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {4, 4, 6, 6, 4096},
                     "surveyType",
                     "3D",
                     "offset_tiles",
                     {}},
        TemplateCase{"OffsetTiles3DDepth",
                     {"inline", "crossline", "inline_offset_tile",
                      "crossline_offset_tile", "depth"},
                     {"cdp_x", "cdp_y"},
                     {},
                     {4, 4, 6, 6, 4096},
                     "surveyType",
                     "3D",
                     "offset_tiles",
                     {}},
        TemplateCase{"StreamerShotGathers2D",
                     {"shot_point", "channel", "time"},
                     {"source_coord_x", "source_coord_y", "group_coord_x",
                      "group_coord_y"},
                     {},
                     {16, 32, 2048},
                     "surveyType",
                     "2D",
                     "common_source",
                     {}},
        TemplateCase{"StreamerShotGathers3D",
                     {"shot_point", "cable", "channel", "time"},
                     {"source_coord_x", "source_coord_y", "group_coord_x",
                      "group_coord_y"},
                     {"gun"},
                     {8, 1, 128, 2048},
                     "surveyType",
                     "3D",
                     "common_source",
                     {}},
        TemplateCase{
            "StreamerFieldRecords3D",
            {"sail_line", "gun", "shot_index", "cable", "channel", "time"},
            {"source_coord_x", "source_coord_y", "group_coord_x",
             "group_coord_y"},
            {"shot_point", "orig_field_record_num"},
            {1, 1, 16, 1, 32, 1024},
            "surveyDimensionality",
            "3D",
            "common_source",
            {"shot_index"}},
        TemplateCase{
            "ObnReceiverGathers3D",
            {"component", "receiver", "shot_line", "gun", "shot_index", "time"},
            {"group_coord_x", "group_coord_y", "source_coord_x",
             "source_coord_y"},
            {"shot_point", "orig_field_record_num"},
            {1, 1, 1, 1, 512, 4096},
            "surveyType",
            "3D",
            "common_receiver",
            {"shot_index"}},
        TemplateCase{"SingleNodeContRecvrGathers",
                     {"component", "epoch", "time"},
                     {"group_coord_x", "group_coord_y"},
                     {},
                     {1, 140, 15001},
                     "surveyType",
                     "3D",
                     "continuous_receiver",
                     {}},
        TemplateCase{
            "ShotReceiverLineGathers3D",
            {"shot_line", "shot_point", "receiver_line", "receiver", "time"},
            {"source_coord_x", "source_coord_y", "group_coord_x",
             "group_coord_y"},
            {"orig_field_record_num"},
            {1, 32, 1, 32, 2048},
            "surveyType",
            "3D",
            "common_source",
            {}}),
    [](const testing::TestParamInfo<TemplateCase>& info) {
      return info.param.name;
    });

TEST(PostStack3D, BuildDatasetVariableShape) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto templ, GetTemplate("PostStack3DTime"));
  TENSORSTORE_ASSERT_OK(
      templ.AddUnits({{"cdp_x", LengthUnit(std::string(mdio::units::kMeters))},
                      {"cdp_y", LengthUnit(std::string(mdio::units::kMeters))},
                      {"time", TimeUnit(std::string(mdio::units::kSeconds))}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto dataset,
      templ.BuildDataset("Seismic 3D", {256, 512, 1024}, StructuredHeaders()));

  const auto* amplitude = FindVariable(dataset, "amplitude");
  ASSERT_NE(amplitude, nullptr);
  EXPECT_EQ((*amplitude)["dataType"], "float32");
  EXPECT_EQ((*amplitude)["compressor"]["cname"], "zstd");
  EXPECT_EQ((*amplitude)["coordinates"], (nlohmann::json{"cdp_x", "cdp_y"}));
  EXPECT_EQ(
      (*amplitude)["metadata"]["chunkGrid"]["configuration"]["chunkShape"],
      (nlohmann::json{128, 128, 128}));

  const auto* headers = FindVariable(dataset, "headers");
  ASSERT_NE(headers, nullptr);
  EXPECT_EQ((*headers)["dimensions"].size(), 2);
  EXPECT_EQ((*headers)["dataType"]["fields"].size(), 4);

  const auto* inline_coord = FindVariable(dataset, "inline");
  ASSERT_NE(inline_coord, nullptr);
  EXPECT_EQ((*inline_coord)["dataType"], "int32");

  const auto* time = FindVariable(dataset, "time");
  ASSERT_NE(time, nullptr);
  EXPECT_EQ((*time)["metadata"]["unitsV1"]["time"], "s");
}

TEST(Coca, AzimuthIsFloat32) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto templ,
                                   GetTemplate("CocaGathers3DTime"));
  EXPECT_EQ(templ.dim_coordinate_types().at("azimuth"), ScalarType::kFloat32);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto dataset,
                                   templ.BuildDataset("coca", {4, 4, 4, 2, 8}));
  const auto* azimuth = FindVariable(dataset, "azimuth");
  ASSERT_NE(azimuth, nullptr);
  EXPECT_EQ((*azimuth)["dataType"], "float32");
}

TEST(StreamerShot3D, GunIsUint8OnShotPoint) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto templ,
                                   GetTemplate("StreamerShotGathers3D"));
  const auto specs = templ.coordinate_specs();
  auto gun = std::find_if(specs.begin(), specs.end(),
                          [](const auto& spec) { return spec.name == "gun"; });
  ASSERT_NE(gun, specs.end());
  EXPECT_EQ(gun->dtype, ScalarType::kUint8);
  EXPECT_EQ(gun->dimensions, (std::vector<std::string>{"shot_point"}));
}

TEST(FieldRecords, OmitsShotIndexDimensionCoordinate) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto templ,
                                   GetTemplate("StreamerFieldRecords3D"));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto dataset, templ.BuildDataset("field", {1, 2, 4, 2, 4, 8}));
  EXPECT_EQ(FindVariable(dataset, "shot_index"), nullptr);
  ASSERT_NE(FindVariable(dataset, "shot_point"), nullptr);
  ASSERT_NE(FindVariable(dataset, "orig_field_record_num"), nullptr);
}

TEST(PostStack3D, FromJsonCreatesDataset) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto templ, GetTemplate("PostStack3DTime"));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema,
                                   templ.BuildDataset("from_json", {8, 8, 16}));
  const std::string path =
      (std::filesystem::temp_directory_path() / "mdio_builder_poststack3d")
          .string();
  std::filesystem::remove_all(path);
  auto future =
      mdio::Dataset::from_json(schema, path, mdio::constants::kCreateClean);
  ASSERT_TRUE(future.result().ok()) << future.result().status();
  std::filesystem::remove_all(path);
}

}  // namespace
