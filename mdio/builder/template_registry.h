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

#ifndef MDIO_BUILDER_TEMPLATE_REGISTRY_H_
#define MDIO_BUILDER_TEMPLATE_REGISTRY_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "mdio/builder/templates/base.h"
#include "mdio/builder/templates/canonical.h"
#include "mdio/impl.h"

namespace mdio {
namespace builder {

/**
 * @brief Thread-safe singleton registry of MDIO dataset templates.
 *
 * Stores `TemplateSpec` records. `Get` returns a `DatasetTemplate` working
 * copy so callers can mutate units / chunks without touching the registry.
 *
 * Comes pre-populated with the 23 canonical seismic templates.
 */
class TemplateRegistry {
 public:
  static TemplateRegistry& GetInstance() {
    static TemplateRegistry instance;
    return instance;
  }

  Result<std::string> Register(TemplateSpec spec) {
    if (spec.name.empty()) {
      return absl::InvalidArgumentError(
          "Cannot register a template with an empty name");
    }
    absl::MutexLock lock(&mutex_);
    if (templates_.count(spec.name) != 0) {
      return absl::AlreadyExistsError(
          absl::StrCat("Template '", spec.name, "' is already registered."));
    }
    std::string name = spec.name;
    templates_.emplace(name, std::move(spec));
    return name;
  }

  Result<DatasetTemplate> Get(const std::string& template_name) const {
    absl::MutexLock lock(&mutex_);
    const auto it = templates_.find(template_name);
    if (it == templates_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Template '", template_name, "' is not registered."));
    }
    return DatasetTemplate(it->second);
  }

  Result<void> Unregister(const std::string& template_name) {
    absl::MutexLock lock(&mutex_);
    const auto it = templates_.find(template_name);
    if (it == templates_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Template '", template_name, "' is not registered."));
    }
    templates_.erase(it);
    return absl::OkStatus();
  }

  bool IsRegistered(const std::string& template_name) const {
    absl::MutexLock lock(&mutex_);
    return templates_.count(template_name) != 0;
  }

  std::vector<std::string> ListAllTemplates() const {
    absl::MutexLock lock(&mutex_);
    std::vector<std::string> names;
    names.reserve(templates_.size());
    for (const auto& [name, _] : templates_) {
      names.push_back(name);
    }
    return names;
  }

  void Clear() {
    absl::MutexLock lock(&mutex_);
    templates_.clear();
  }

  /**
   * @brief Installs any missing built-in specs. Does not overwrite custom
   * entries already present under the same name.
   */
  void RegisterDefaultTemplates() {
    absl::MutexLock lock(&mutex_);
    InstallCanonicalSpecs();
  }

  /**
   * @brief Testing helper: restore the 23 built-in specs.
   */
  static void ResetInstanceForTesting() {
    TemplateRegistry& instance = GetInstance();
    absl::MutexLock lock(&instance.mutex_);
    instance.templates_.clear();
    instance.InstallCanonicalSpecs();
  }

 private:
  TemplateRegistry() {
    absl::MutexLock lock(&mutex_);
    InstallCanonicalSpecs();
  }

  TemplateRegistry(const TemplateRegistry&) = delete;
  TemplateRegistry& operator=(const TemplateRegistry&) = delete;

  void InstallCanonicalSpecs() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    for (auto spec : CanonicalTemplateSpecs()) {
      templates_.emplace(spec.name, std::move(spec));
    }
  }

  mutable absl::Mutex mutex_;
  std::unordered_map<std::string, TemplateSpec> templates_
      ABSL_GUARDED_BY(mutex_);
};

inline TemplateRegistry& GetTemplateRegistry() {
  return TemplateRegistry::GetInstance();
}

inline Result<std::string> RegisterTemplate(TemplateSpec spec) {
  return GetTemplateRegistry().Register(std::move(spec));
}

inline Result<DatasetTemplate> GetTemplate(const std::string& name) {
  return GetTemplateRegistry().Get(name);
}

inline bool IsTemplateRegistered(const std::string& name) {
  return GetTemplateRegistry().IsRegistered(name);
}

inline std::vector<std::string> ListTemplates() {
  return GetTemplateRegistry().ListAllTemplates();
}

}  // namespace builder
}  // namespace mdio

#endif  // MDIO_BUILDER_TEMPLATE_REGISTRY_H_
