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

#include <iostream>
#include <string>

#include "mdio/labels/labels.h"

// clang-format off
#include <nlohmann/json.hpp>  // NOLINT
// clang-format on

namespace {

void PrintUsage() {
  std::cerr
      << "Usage: mdio_labels --source <mdio> --annotation <dump.json> "
         "[--dest <mdio>] [--source-variable amplitude] "
         "[--variable-name labels] [--threads 8] [--s3-concurrency 64] "
         "[--cache-mb 1024]\n";
}

bool ConsumeFlag(int& i, int argc, char** argv, const char* name,
                 std::string* value) {
  const std::string arg = argv[i];
  const std::string prefix = std::string(name) + "=";
  if (arg == name) {
    if (i + 1 >= argc) {
      return false;
    }
    *value = argv[++i];
    return true;
  }
  if (arg.rfind(prefix, 0) == 0) {
    *value = arg.substr(prefix.size());
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  mdio::labels::WriteLabelsConfig config;
  std::string annotation_path;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      PrintUsage();
      return 0;
    }
    std::string value;
    if (ConsumeFlag(i, argc, argv, "--source", &value)) {
      config.source_path = value;
    } else if (ConsumeFlag(i, argc, argv, "--dest", &value)) {
      config.dest_path = value;
    } else if (ConsumeFlag(i, argc, argv, "--annotation", &value)) {
      annotation_path = value;
    } else if (ConsumeFlag(i, argc, argv, "--source-variable", &value)) {
      config.source_variable = value;
    } else if (ConsumeFlag(i, argc, argv, "--variable-name", &value)) {
      config.variable_name = value;
    } else if (ConsumeFlag(i, argc, argv, "--threads", &value)) {
      config.threads = std::stoi(value);
    } else if (ConsumeFlag(i, argc, argv, "--s3-concurrency", &value)) {
      config.s3_concurrency = std::stoi(value);
    } else if (ConsumeFlag(i, argc, argv, "--cache-mb", &value)) {
      config.cache_mb = std::stoi(value);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage();
      return 2;
    }
  }

  if (config.source_path.empty() || annotation_path.empty()) {
    PrintUsage();
    return 2;
  }

  config.progress = [](int64_t done, int64_t total) {
    nlohmann::json line = {
        {"event", "progress"}, {"done", done}, {"total", total}};
    std::cout << line.dump() << std::endl;
  };

  auto dump = mdio::labels::LoadAnnotationDump(annotation_path);
  if (!dump.status().ok()) {
    std::cerr << dump.status() << "\n";
    return 1;
  }
  auto result = mdio::labels::WriteLabels(config, dump.value());
  if (!result.status().ok()) {
    std::cerr << result.status() << "\n";
    return 1;
  }

  nlohmann::json done = {{"event", "done"},
                         {"variable", result->variable_name},
                         {"labelSize", result->label_size},
                         {"dest", result->dest_path}};
  std::cout << done.dump() << std::endl;
  return 0;
}
