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

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mdio/optimize/access_pattern.h"

// clang-format off
#include <nlohmann/json.hpp>  // NOLINT
// clang-format on

namespace {

void PrintUsage() {
  std::cerr
      << "Usage: mdio_optimize --path <mdio> --mags 2,4,8 "
         "[--max-memory-mb 2048] [--threads 8] [--source amplitude] "
         "[--s3-concurrency 64] [--cache-mb 1024]\n";
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

std::vector<int> ParseMags(const std::string& raw) {
  std::vector<int> mags;
  std::stringstream stream(raw);
  std::string part;
  while (std::getline(stream, part, ',')) {
    if (part.empty()) {
      continue;
    }
    mags.push_back(std::stoi(part));
  }
  return mags;
}

int MagFromName(const std::string& name) {
  const std::string prefix = "fast_mag_";
  if (name.rfind(prefix, 0) != 0) {
    return 0;
  }
  try {
    return std::stoi(name.substr(prefix.size()));
  } catch (...) {
    return 0;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  std::string mags_raw;
  std::string source;
  int max_memory_mb = 2048;
  int threads = 8;
  int s3_concurrency = 64;
  int cache_mb = 1024;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      PrintUsage();
      return 0;
    }
    std::string value;
    if (ConsumeFlag(i, argc, argv, "--path", &value)) {
      path = value;
    } else if (ConsumeFlag(i, argc, argv, "--mags", &value)) {
      mags_raw = value;
    } else if (ConsumeFlag(i, argc, argv, "--source", &value)) {
      source = value;
    } else if (ConsumeFlag(i, argc, argv, "--max-memory-mb", &value)) {
      max_memory_mb = std::stoi(value);
    } else if (ConsumeFlag(i, argc, argv, "--threads", &value)) {
      threads = std::stoi(value);
    } else if (ConsumeFlag(i, argc, argv, "--s3-concurrency", &value)) {
      s3_concurrency = std::stoi(value);
    } else if (ConsumeFlag(i, argc, argv, "--cache-mb", &value)) {
      cache_mb = std::stoi(value);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage();
      return 2;
    }
  }

  if (path.empty() || mags_raw.empty()) {
    PrintUsage();
    return 2;
  }
  if (max_memory_mb <= 0 || threads < 0 || s3_concurrency <= 0 ||
      cache_mb <= 0) {
    std::cerr << "--max-memory-mb, --s3-concurrency, and --cache-mb must be "
                 "> 0; --threads must be >= 0\n";
    return 2;
  }

  std::vector<int> mags;
  try {
    mags = ParseMags(mags_raw);
  } catch (...) {
    std::cerr << "Invalid --mags value: " << mags_raw << "\n";
    return 2;
  }
  if (mags.empty()) {
    std::cerr << "--mags must list at least one factor\n";
    return 2;
  }

  mdio::optimize::OptimizedAccessPatternConfig config;
  config.magnification_factors = mags;
  config.source_variable = source;
  config.max_memory_bytes = static_cast<size_t>(max_memory_mb) * 1024ull * 1024ull;
  config.max_threads = threads;

  std::mutex out_mu;
  config.progress = [&](const std::string& name, int64_t done, int64_t total) {
    const int mag = MagFromName(name);
    if (mag < 2) {
      return;
    }
    nlohmann::json line = {{"event", "progress"},
                           {"mag", mag},
                           {"done", done},
                           {"total", total}};
    std::lock_guard<std::mutex> lock(out_mu);
    std::cout << line.dump() << std::endl;
  };

  const int copy_limit =
      threads > 0 ? threads
                  : static_cast<int>(std::thread::hardware_concurrency());
  nlohmann::json ctx_json = {
      {"cache_pool",
       {{"total_bytes_limit",
         static_cast<uint64_t>(cache_mb) * 1024ull * 1024ull}}},
      {"data_copy_concurrency", {{"limit", copy_limit > 0 ? copy_limit : 32}}},
      {"s3_request_concurrency", {{"limit", s3_concurrency}}}};
  auto spec = tensorstore::Context::Spec::FromJson(ctx_json);
  if (!spec.ok()) {
    ctx_json = {{"cache_pool",
                 {{"total_bytes_limit",
                   static_cast<uint64_t>(cache_mb) * 1024ull * 1024ull}}}};
    spec = tensorstore::Context::Spec::FromJson(ctx_json);
  }
  if (!spec.ok()) {
    std::cerr << "context spec failed: " << spec.status() << "\n";
    return 1;
  }
  tensorstore::Context ctx(*spec);

  auto result = mdio::optimize::OptimizeAccessPatterns(path, config, ctx);
  if (!result.status().ok()) {
    std::cerr << result.status() << std::endl;
    return 1;
  }

  // Do not Dataset::Open the whole store. FAP siblings use zfpy.
  std::vector<int> written = mags;

  nlohmann::json done = {{"event", "done"}, {"mags", written}};
  std::cout << done.dump() << std::endl;
  return 0;
}
