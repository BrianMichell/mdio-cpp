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

#ifndef PYTHON_SRC_BINDINGS_STATUS_H_
#define PYTHON_SRC_BINDINGS_STATUS_H_

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "mdio/impl.h"

namespace py = pybind11;

namespace mdio_py {

class MdioError : public std::runtime_error {
 public:
  explicit MdioError(const std::string& message)
      : std::runtime_error(message) {}
};

inline void ThrowIfError(const absl::Status& status) {
  if (!status.ok()) {
    throw MdioError(std::string(status.ToString()));
  }
}

template <typename T>
T CheckResult(mdio::Result<T> result) {
  ThrowIfError(result.status());
  return std::move(result).value();
}

inline void CheckResult(const mdio::Result<void>& result) {
  ThrowIfError(result.status());
}

namespace detail {
template <typename>
struct IsFuture : std::false_type {};
template <typename T>
struct IsFuture<mdio::Future<T>> : std::true_type {};
template <typename T>
struct FutureValue;
template <typename T>
struct FutureValue<mdio::Future<T>> {
  using type = T;
};

template <typename>
struct IsResult : std::false_type {};
template <typename T>
struct IsResult<mdio::Result<T>> : std::true_type {};
}  // namespace detail

// One I/O door: drop GIL, run fn, wait if it returned a Future / WriteFutures.
// Callers do not need to know whether C++ waits before or after the Future.
template <typename F>
auto Await(F&& fn) {
  using Out = std::decay_t<decltype(fn())>;
  if constexpr (detail::IsFuture<Out>::value) {
    using T = typename detail::FutureValue<Out>::type;
    if constexpr (std::is_void_v<T>) {
      ThrowIfError([&] {
        py::gil_scoped_release release;
        return std::forward<F>(fn)().status();
      }());
    } else {
      return CheckResult([&] {
        py::gil_scoped_release release;
        return std::forward<F>(fn)().result();
      }());
    }
  } else if constexpr (detail::IsResult<Out>::value) {
    return CheckResult([&] {
      py::gil_scoped_release release;
      return std::forward<F>(fn)();
    }());
  } else if constexpr (std::is_same_v<Out, mdio::WriteFutures>) {
    ThrowIfError([&] {
      py::gil_scoped_release release;
      return std::forward<F>(fn)().status();
    }());
  } else {
    static_assert(!sizeof(Out),
                  "Await expects Future, Result, or WriteFutures");
  }
}

}  // namespace mdio_py

#endif  // PYTHON_SRC_BINDINGS_STATUS_H_
