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

#ifndef MDIO_BUILDER_TEMPLATES_CANONICAL_H_
#define MDIO_BUILDER_TEMPLATES_CANONICAL_H_

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mdio/builder/templates/types.h"

namespace mdio {
namespace builder {

inline TemplateSpec PostStack2DSpec(SeismicDataDomain domain) {
  const char* d = ToString(domain);
  TemplateSpec spec;
  spec.name = absl::StrCat("PostStack2D", AsciiCapitalize(d));
  spec.data_domain = domain;
  spec.dims = {{"cdp"}, {d}};
  spec.coords = {{"cdp_x", {"cdp"}}, {"cdp_y", {"cdp"}}};
  spec.chunks = {1024, 1024};
  spec.attributes = {{"surveyType", "2D"}, {"gatherType", "stacked"}};
  return spec;
}

inline TemplateSpec PostStack3DSpec(SeismicDataDomain domain) {
  const char* d = ToString(domain);
  TemplateSpec spec;
  spec.name = absl::StrCat("PostStack3D", AsciiCapitalize(d));
  spec.data_domain = domain;
  spec.dims = {{"inline"}, {"crossline"}, {d}};
  spec.coords = {{"cdp_x", {"inline", "crossline"}},
                 {"cdp_y", {"inline", "crossline"}}};
  spec.chunks = {128, 128, 128};
  spec.attributes = {{"surveyType", "3D"}, {"gatherType", "stacked"}};
  return spec;
}

inline TemplateSpec CdpGathers2DSpec(SeismicDataDomain domain,
                                     CdpGatherDomain gather) {
  const char* d = ToString(domain);
  const char* g = ToString(gather);
  TemplateSpec spec;
  spec.name =
      absl::StrCat("Cdp", AsciiCapitalize(g), "Gathers2D", AsciiCapitalize(d));
  spec.data_domain = domain;
  spec.dims = {{"cdp"}, {g}, {d}};
  spec.coords = {{"cdp_x", {"cdp"}}, {"cdp_y", {"cdp"}}};
  spec.chunks = {16, 64, 1024};
  spec.attributes = {{"surveyType", "2D"}, {"gatherType", "cdp"}};
  return spec;
}

inline TemplateSpec CdpGathers3DSpec(SeismicDataDomain domain,
                                     CdpGatherDomain gather) {
  const char* d = ToString(domain);
  const char* g = ToString(gather);
  TemplateSpec spec;
  spec.name =
      absl::StrCat("Cdp", AsciiCapitalize(g), "Gathers3D", AsciiCapitalize(d));
  spec.data_domain = domain;
  spec.dims = {{"inline"}, {"crossline"}, {g}, {d}};
  spec.coords = {{"cdp_x", {"inline", "crossline"}},
                 {"cdp_y", {"inline", "crossline"}}};
  spec.chunks = {8, 8, 32, 512};
  spec.attributes = {{"surveyType", "3D"}, {"gatherType", "cdp"}};
  return spec;
}

inline TemplateSpec CocaGathers3DSpec(SeismicDataDomain domain) {
  const char* d = ToString(domain);
  TemplateSpec spec;
  spec.name = absl::StrCat("CocaGathers3D", AsciiCapitalize(d));
  spec.data_domain = domain;
  spec.dims = {{"inline"},
               {"crossline"},
               {"offset"},
               {"azimuth", ScalarType::kFloat32},
               {d}};
  spec.coords = {{"cdp_x", {"inline", "crossline"}},
                 {"cdp_y", {"inline", "crossline"}}};
  spec.chunks = {8, 8, 32, 1, 1024};
  spec.attributes = {{"surveyType", "3D"},
                     {"gatherType", "common_offset_common_azimuth"}};
  return spec;
}

inline TemplateSpec ReceiverGathers3DSpec() {
  TemplateSpec spec;
  spec.name = "ReceiverGathers3D";
  spec.dims = {{"receiver", ScalarType::kUint32},
               {"shot_line", ScalarType::kUint32},
               {"shot_point", ScalarType::kUint32},
               {"time"}};
  spec.coords = {
      {"receiver_x", {"receiver"}},
      {"receiver_y", {"receiver"}},
      {"source_coord_x", {"shot_line", "shot_point"}},
      {"source_coord_y", {"shot_line", "shot_point"}},
  };
  spec.chunks = {1, 1, 512, 4096};
  spec.attributes = {{"surveyType", "3D"}, {"gatherType", "receiver_gathers"}};
  return spec;
}

inline TemplateSpec OffsetTiles3DSpec(SeismicDataDomain domain) {
  const char* d = ToString(domain);
  TemplateSpec spec;
  spec.name = absl::StrCat("OffsetTiles3D", AsciiCapitalize(d));
  spec.data_domain = domain;
  spec.dims = {{"inline"},
               {"crossline"},
               {"inline_offset_tile", ScalarType::kInt16},
               {"crossline_offset_tile", ScalarType::kInt16},
               {d}};
  spec.coords = {{"cdp_x", {"inline", "crossline"}},
                 {"cdp_y", {"inline", "crossline"}}};
  spec.chunks = {4, 4, 6, 6, 4096};
  spec.attributes = {{"surveyType", "3D"}, {"gatherType", "offset_tiles"}};
  return spec;
}

inline TemplateSpec StreamerShotGathers2DSpec() {
  TemplateSpec spec;
  spec.name = "StreamerShotGathers2D";
  spec.dims = {{"shot_point"}, {"channel"}, {"time"}};
  spec.coords = {
      {"source_coord_x", {"shot_point"}},
      {"source_coord_y", {"shot_point"}},
      {"group_coord_x", {"shot_point", "channel"}},
      {"group_coord_y", {"shot_point", "channel"}},
  };
  spec.chunks = {16, 32, 2048};
  spec.attributes = {{"surveyType", "2D"}, {"gatherType", "common_source"}};
  return spec;
}

inline TemplateSpec StreamerShotGathers3DSpec() {
  TemplateSpec spec;
  spec.name = "StreamerShotGathers3D";
  spec.dims = {{"shot_point"}, {"cable"}, {"channel"}, {"time"}};
  spec.coords = {
      {"source_coord_x", {"shot_point"}},
      {"source_coord_y", {"shot_point"}},
      {"group_coord_x", {"shot_point", "cable", "channel"}},
      {"group_coord_y", {"shot_point", "cable", "channel"}},
      {"gun", {"shot_point"}, ScalarType::kUint8, CoordRole::kLogical},
  };
  spec.chunks = {8, 1, 128, 2048};
  spec.attributes = {{"surveyType", "3D"}, {"gatherType", "common_source"}};
  return spec;
}

inline TemplateSpec StreamerFieldRecords3DSpec() {
  const std::vector<std::string> shot = {"sail_line", "gun", "shot_index"};
  const std::vector<std::string> recv = {"sail_line", "gun", "shot_index",
                                         "cable", "channel"};
  TemplateSpec spec;
  spec.name = "StreamerFieldRecords3D";
  spec.dims = {{"sail_line", ScalarType::kUint32},
               {"gun", ScalarType::kUint8},
               {"shot_index", ScalarType::kInt32, DimKind::kCalculated},
               {"cable", ScalarType::kUint8},
               {"channel", ScalarType::kUint16},
               {"time"}};
  spec.coords = {
      {"source_coord_x", shot},
      {"source_coord_y", shot},
      {"group_coord_x", recv},
      {"group_coord_y", recv},
      {"shot_point", shot, ScalarType::kUint32, CoordRole::kLogical},
      {"orig_field_record_num", shot, ScalarType::kUint32, CoordRole::kLogical},
  };
  spec.chunks = {1, 1, 16, 1, 32, 1024};
  spec.attributes = {{"surveyDimensionality", "3D"},
                     {"gatherType", "common_source"}};
  spec.blosc_on_non_dim_coords = false;
  return spec;
}

inline TemplateSpec ObnReceiverGathers3DSpec() {
  const std::vector<std::string> recv = {"receiver"};
  const std::vector<std::string> shot = {"shot_line", "gun", "shot_index"};
  TemplateSpec spec;
  spec.name = "ObnReceiverGathers3D";
  spec.dims = {{"component", ScalarType::kUint8, DimKind::kSynthesizeIfMissing},
               {"receiver", ScalarType::kUint32},
               {"shot_line", ScalarType::kUint32},
               {"gun", ScalarType::kUint8},
               {"shot_index", ScalarType::kInt32, DimKind::kCalculated},
               {"time"}};
  spec.coords = {
      {"group_coord_x", recv},
      {"group_coord_y", recv},
      {"source_coord_x", shot},
      {"source_coord_y", shot},
      {"shot_point", shot, ScalarType::kUint32, CoordRole::kLogical},
      {"orig_field_record_num", shot, ScalarType::kUint32, CoordRole::kLogical},
  };
  spec.chunks = {1, 1, 1, 1, 512, 4096};
  spec.attributes = {{"surveyType", "3D"}, {"gatherType", "common_receiver"}};
  spec.blosc_on_non_dim_coords = false;
  return spec;
}

inline TemplateSpec SingleNodeContRecvrGathersSpec() {
  TemplateSpec spec;
  spec.name = "SingleNodeContRecvrGathers";
  spec.dims = {{"component", ScalarType::kUint8, DimKind::kSynthesizeIfMissing},
               {"epoch", ScalarType::kInt64},
               {"time"}};
  spec.coords = {{"group_coord_x", {"component"}},
                 {"group_coord_y", {"component"}}};
  spec.chunks = {1, 140, 15001};
  spec.attributes = {{"surveyType", "3D"},
                     {"gatherType", "continuous_receiver"}};
  spec.default_units = {
      {"epoch", TimeUnit(std::string(mdio::units::kMicroseconds))}};
  spec.blosc_on_non_dim_coords = false;
  return spec;
}

inline TemplateSpec ShotReceiverLineGathers3DSpec() {
  const std::vector<std::string> source = {"shot_line", "shot_point"};
  const std::vector<std::string> group = {"receiver_line", "receiver"};
  TemplateSpec spec;
  spec.name = "ShotReceiverLineGathers3D";
  spec.dims = {{"shot_line", ScalarType::kUint32},
               {"shot_point", ScalarType::kUint32},
               {"receiver_line", ScalarType::kUint32},
               {"receiver", ScalarType::kUint32},
               {"time"}};
  spec.coords = {
      {"source_coord_x", source},
      {"source_coord_y", source},
      {"group_coord_x", group},
      {"group_coord_y", group},
      {"orig_field_record_num", source, ScalarType::kUint32,
       CoordRole::kLogical},
  };
  spec.chunks = {1, 32, 1, 32, 2048};
  spec.attributes = {{"surveyType", "3D"}, {"gatherType", "common_source"}};
  spec.blosc_on_non_dim_coords = false;
  return spec;
}

/**
 * @brief The 23 canonical seismic templates, in Python registration order.
 */
inline std::vector<TemplateSpec> CanonicalTemplateSpecs() {
  std::vector<TemplateSpec> specs;
  specs.reserve(23);
  specs.push_back(PostStack2DSpec(SeismicDataDomain::kTime));
  specs.push_back(PostStack2DSpec(SeismicDataDomain::kDepth));
  specs.push_back(PostStack3DSpec(SeismicDataDomain::kTime));
  specs.push_back(PostStack3DSpec(SeismicDataDomain::kDepth));
  for (SeismicDataDomain data_domain :
       {SeismicDataDomain::kTime, SeismicDataDomain::kDepth}) {
    for (CdpGatherDomain gather_domain :
         {CdpGatherDomain::kOffset, CdpGatherDomain::kAngle}) {
      specs.push_back(CdpGathers3DSpec(data_domain, gather_domain));
      specs.push_back(CdpGathers2DSpec(data_domain, gather_domain));
    }
  }
  specs.push_back(CocaGathers3DSpec(SeismicDataDomain::kTime));
  specs.push_back(CocaGathers3DSpec(SeismicDataDomain::kDepth));
  specs.push_back(ReceiverGathers3DSpec());
  specs.push_back(OffsetTiles3DSpec(SeismicDataDomain::kTime));
  specs.push_back(OffsetTiles3DSpec(SeismicDataDomain::kDepth));
  specs.push_back(StreamerShotGathers2DSpec());
  specs.push_back(StreamerShotGathers3DSpec());
  specs.push_back(StreamerFieldRecords3DSpec());
  specs.push_back(ObnReceiverGathers3DSpec());
  specs.push_back(SingleNodeContRecvrGathersSpec());
  specs.push_back(ShotReceiverLineGathers3DSpec());
  return specs;
}

}  // namespace builder
}  // namespace mdio

#endif  // MDIO_BUILDER_TEMPLATES_CANONICAL_H_
