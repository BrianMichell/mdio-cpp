// Copyright 2026 TGS
//
// Persistent MDIO read sidecar for WEBKNOSSOS. Play talks TCP localhost.
// TensorStore owns S3 + decode. Java zarr path is not used.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mdio/dataset.h"
#include "mdio/zarr/zarr_driver.h"

// clang-format off
#include <nlohmann/json.hpp>  // NOLINT
// clang-format on

namespace {

constexpr uint32_t kMaxJsonBytes = 16 * 1024 * 1024;

struct Item {
  int64_t ox = 0;
  int64_t oy = 0;
  int64_t oz = 0;
  int64_t sx = 32;
  int64_t sy = 32;
  int64_t sz = 32;
  int64_t mx = 1;
  int64_t my = 1;
  int64_t mz = 1;
};

struct Request {
  std::string dataset;
  std::string variable;
  std::string x_dim;
  std::string y_dim;
  std::string z_dim;
  int x_idx = -1;
  int y_idx = -1;
  int z_idx = -1;
  std::vector<Item> items;
};

bool ReadExact(int fd, void* buf, size_t n) {
  auto* p = static_cast<char*>(buf);
  size_t got = 0;
  while (got < n) {
    const ssize_t r = ::read(fd, p + got, n - got);
    if (r <= 0) {
      return false;
    }
    got += static_cast<size_t>(r);
  }
  return true;
}

bool WriteExact(int fd, const void* buf, size_t n) {
  auto* p = static_cast<const char*>(buf);
  size_t sent = 0;
  while (sent < n) {
    const ssize_t w = ::write(fd, p + sent, n - sent);
    if (w <= 0) {
      return false;
    }
    sent += static_cast<size_t>(w);
  }
  return true;
}

bool ReadFrame(int fd, std::string* json) {
  uint32_t le = 0;
  if (!ReadExact(fd, &le, 4)) {
    return false;
  }
  const uint32_t n = le;
  if (n == 0 || n > kMaxJsonBytes) {
    return false;
  }
  json->assign(n, '\0');
  return ReadExact(fd, json->data(), n);
}

bool WriteFrame(int fd, const std::string& json, const std::string& payload) {
  const uint32_t n = static_cast<uint32_t>(json.size());
  if (!WriteExact(fd, &n, 4) || !WriteExact(fd, json.data(), json.size())) {
    return false;
  }
  if (payload.empty()) {
    return true;
  }
  return WriteExact(fd, payload.data(), payload.size());
}

mdio::Result<Request> ParseRequest(const nlohmann::json& j) {
  Request req;
  if (!j.contains("dataset") || !j.contains("variable")) {
    return absl::InvalidArgumentError("dataset and variable required");
  }
  req.dataset = j["dataset"].get<std::string>();
  req.variable = j["variable"].get<std::string>();
  req.x_dim = j.value("xDim", "crossline");
  req.y_dim = j.value("yDim", "inline");
  req.z_dim = j.value("zDim", "depth");
  req.x_idx = j.value("xIdx", -1);
  req.y_idx = j.value("yIdx", -1);
  req.z_idx = j.value("zIdx", -1);
  if (!j.contains("items") || !j["items"].is_array()) {
    return absl::InvalidArgumentError("items array required");
  }
  for (const auto& it : j["items"]) {
    Item item;
    item.ox = it.value("ox", 0);
    item.oy = it.value("oy", 0);
    item.oz = it.value("oz", 0);
    item.sx = it.value("sx", 32);
    item.sy = it.value("sy", 32);
    item.sz = it.value("sz", 32);
    item.mx = std::max<int64_t>(1, it.value("mx", 1));
    item.my = std::max<int64_t>(1, it.value("my", 1));
    item.mz = std::max<int64_t>(1, it.value("mz", 1));
    req.items.push_back(item);
  }
  return req;
}

int64_t Clamp(int64_t v, int64_t lo, int64_t hi) {
  return std::max(lo, std::min(v, hi));
}

mdio::Result<int> LabelIndex(const mdio::Variable<>& var,
                             const std::string& name) {
  const auto labels = var.dimensions().labels();
  for (mdio::DimensionIndex i = 0; i < var.dimensions().rank(); ++i) {
    if (std::string(labels[i]) == name) {
      return static_cast<int>(i);
    }
  }
  return absl::NotFoundError("dimension not found: " + name);
}

const char* OriginBytes(mdio::VariableData<>& data) {
  auto acc = data.get_data_accessor();
  return reinterpret_cast<const char*>(
      acc.byte_strided_origin_pointer().get());
}

void CopyToFortranXyz(mdio::VariableData<>& data, int ix, int iy, int iz,
                      const Item& item, int64_t pad_x, int64_t pad_y,
                      int64_t pad_z, std::string* out) {
  auto acc = data.get_data_accessor();
  const int elem = static_cast<int>(data.dtype().size());
  const auto strides = acc.byte_strides();
  const char* base = OriginBytes(data);
  const int64_t sx = item.sx;
  const int64_t sy = item.sy;
  const int64_t sz = item.sz;
  out->assign(static_cast<size_t>(sx * sy * sz * elem), '\0');
  char* dst = out->data();
  const auto shape = data.dimensions().shape();
  const int64_t nx = shape[ix];
  const int64_t ny = shape[iy];
  const int64_t nz = iz >= 0 ? shape[iz] : 1;
  for (int64_t z = 0; z < sz; ++z) {
    const int64_t zs = z - pad_z;
    if (zs < 0 || zs >= nz) {
      continue;
    }
    for (int64_t y = 0; y < sy; ++y) {
      const int64_t ys = y - pad_y;
      if (ys < 0 || ys >= ny) {
        continue;
      }
      for (int64_t x = 0; x < sx; ++x) {
        const int64_t xs = x - pad_x;
        if (xs < 0 || xs >= nx) {
          continue;
        }
        int64_t off = xs * strides[ix] + ys * strides[iy];
        if (iz >= 0) {
          off += zs * strides[iz];
        }
        const int64_t dst_i = (x + y * sx + z * sx * sy) * elem;
        std::memcpy(dst + dst_i, base + off, static_cast<size_t>(elem));
      }
    }
  }
}

template <typename T>
void AverageToFortranXyz(mdio::VariableData<>& data, int ix, int iy, int iz,
                         const Item& item, int64_t pad_x, int64_t pad_y,
                         int64_t pad_z, std::string* out) {
  auto acc = data.get_data_accessor();
  const auto strides = acc.byte_strides();
  const char* base = OriginBytes(data);
  const int64_t sx = item.sx;
  const int64_t sy = item.sy;
  const int64_t sz = item.sz;
  out->assign(static_cast<size_t>(sx * sy * sz * sizeof(T)), '\0');
  T* dst = reinterpret_cast<T*>(out->data());
  const auto shape = data.dimensions().shape();
  const int64_t nx = shape[ix];
  const int64_t ny = shape[iy];
  const int64_t nz = iz >= 0 ? shape[iz] : 1;
  for (int64_t oz = 0; oz < sz; ++oz) {
    for (int64_t oy = 0; oy < sy; ++oy) {
      for (int64_t ox = 0; ox < sx; ++ox) {
        double sum = 0;
        int n = 0;
        for (int64_t z = 0; z < item.mz; ++z) {
          const int64_t zz = oz * item.mz + z - pad_z;
          if (zz < 0 || zz >= nz) {
            continue;
          }
          for (int64_t y = 0; y < item.my; ++y) {
            const int64_t yy = oy * item.my + y - pad_y;
            if (yy < 0 || yy >= ny) {
              continue;
            }
            for (int64_t x = 0; x < item.mx; ++x) {
              const int64_t xx = ox * item.mx + x - pad_x;
              if (xx < 0 || xx >= nx) {
                continue;
              }
              int64_t off = xx * strides[ix] + yy * strides[iy];
              if (iz >= 0) {
                off += zz * strides[iz];
              }
              T v;
              std::memcpy(&v, base + off, sizeof(T));
              if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(static_cast<double>(v))) {
                  continue;
                }
              }
              sum += static_cast<double>(v);
              ++n;
            }
          }
        }
        dst[ox + oy * sx + oz * sx * sy] =
            n == 0 ? T{0} : static_cast<T>(sum / n);
      }
    }
  }
}

struct Inflight {
  Item item;
  int six = 0;
  int siy = 1;
  int siz = -1;
  int64_t pad_x = 0;
  int64_t pad_y = 0;
  int64_t pad_z = 0;
  std::optional<mdio::Variable<>> sliced;
  std::optional<mdio::Future<mdio::VariableData<>>> fut;
};

const char* kZAliases[] = {"depth", "time",  "sample", "twt",
                           "tvd",   "tvdss", "z",      nullptr};
const char* kXAliases[] = {"crossline", "xline", "cdp-x", "cdp_x", "x",
                           nullptr};
const char* kYAliases[] = {"inline", "iline", "cdp-y", "cdp_y", "y", nullptr};

int TryAliases(const mdio::Variable<>& var, const char* const* aliases) {
  for (const char* const* p = aliases; p != nullptr && *p != nullptr; ++p) {
    auto found = LabelIndex(var, *p);
    if (found.ok()) {
      return found.value();
    }
  }
  return -1;
}

int ResolveDim(const mdio::Variable<>& var, const std::string& name, int idx,
               const char* const* aliases) {
  if (!name.empty()) {
    auto found = LabelIndex(var, name);
    if (found.ok()) {
      return found.value();
    }
  }
  const int aliased = TryAliases(var, aliases);
  if (aliased >= 0) {
    return aliased;
  }
  const int rank = static_cast<int>(var.dimensions().rank());
  if (idx >= 0 && idx < rank) {
    return idx;
  }
  return -1;
}

std::string LabelAt(const mdio::Variable<>& var, int idx) {
  if (idx < 0) {
    return "";
  }
  return std::string(var.dimensions().labels()[idx]);
}

mdio::RangeDescriptor<mdio::Index> DimSlice(const std::string& name, int idx,
                                            int64_t start, int64_t stop) {
  if (!name.empty()) {
    return {name, start, stop, 1};
  }
  return {static_cast<mdio::DimensionIndex>(idx), start, stop, 1};
}

void FillDimOrder(const mdio::Variable<>& sliced, const std::string& x_name,
                  const std::string& y_name, const std::string& z_name, int iz,
                  Inflight* out) {
  const auto sl = sliced.dimensions().labels();
  for (mdio::DimensionIndex i = 0; i < sliced.dimensions().rank(); ++i) {
    const std::string lab(sl[i]);
    if (!x_name.empty() && lab == x_name) {
      out->six = static_cast<int>(i);
    }
    if (!y_name.empty() && lab == y_name) {
      out->siy = static_cast<int>(i);
    }
    if (iz >= 0 && !z_name.empty() && lab == z_name) {
      out->siz = static_cast<int>(i);
    }
  }
}

mdio::Result<Inflight> StartRead(mdio::Variable<> var, const Request& req,
                                 const Item& item) {
  Inflight out;
  out.item = item;
  const int ix = ResolveDim(var, req.x_dim, req.x_idx, kXAliases);
  const int iy = ResolveDim(var, req.y_dim, req.y_idx, kYAliases);
  if (ix < 0 || iy < 0) {
    return absl::NotFoundError("x/y dimension not found");
  }
  const int iz = ResolveDim(var, req.z_dim, req.z_idx, kZAliases);
  const auto domain = var.dimensions();
  const std::string x_name = LabelAt(var, ix);
  const std::string y_name = LabelAt(var, iy);
  const std::string z_name = LabelAt(var, iz);
  // WK bbox is 0-based. MDIO slice uses labeled origin (inline numbers, etc).
  const int64_t xo = domain.origin()[ix];
  const int64_t yo = domain.origin()[iy];
  const int64_t x0 = xo + item.ox * item.mx;
  const int64_t y0 = yo + item.oy * item.my;
  const int64_t x1 = x0 + item.sx * item.mx;
  const int64_t y1 = y0 + item.sy * item.my;
  const int64_t xs = domain.shape()[ix];
  const int64_t ys = domain.shape()[iy];
  const int64_t cx0 = Clamp(x0, xo, xo + xs);
  const int64_t cx1 = Clamp(x1, xo, xo + xs);
  const int64_t cy0 = Clamp(y0, yo, yo + ys);
  const int64_t cy1 = Clamp(y1, yo, yo + ys);
  if (cx1 <= cx0 || cy1 <= cy0) {
    return out;
  }
  std::vector<mdio::RangeDescriptor<mdio::Index>> slices;
  slices.push_back(DimSlice(x_name, ix, cx0, cx1));
  slices.push_back(DimSlice(y_name, iy, cy0, cy1));
  int64_t z0 = item.oz * item.mz;
  int64_t cz0 = z0;
  int64_t cz1 = z0 + item.sz * item.mz;
  if (iz >= 0) {
    const int64_t zo = domain.origin()[iz];
    const int64_t zs = domain.shape()[iz];
    z0 = zo + item.oz * item.mz;
    cz0 = Clamp(z0, zo, zo + zs);
    cz1 = Clamp(z0 + item.sz * item.mz, zo, zo + zs);
    if (cz1 <= cz0) {
      return out;
    }
    slices.push_back(DimSlice(z_name, iz, cz0, cz1));
  }
  for (mdio::DimensionIndex i = 0; i < domain.rank(); ++i) {
    const int ii = static_cast<int>(i);
    if (ii == ix || ii == iy || ii == iz) {
      continue;
    }
    const int64_t o = domain.origin()[i];
    slices.push_back(DimSlice(std::string(domain.labels()[i]), ii, o, o + 1));
  }
  out.six = ix;
  out.siy = iy;
  out.siz = iz;
  out.pad_x = cx0 - x0;
  out.pad_y = cy0 - y0;
  out.pad_z = iz >= 0 ? cz0 - z0 : 0;
  MDIO_ASSIGN_OR_RETURN(auto sliced, var.slice(slices));
  FillDimOrder(sliced, x_name, y_name, z_name, iz, &out);
  out.sliced = std::move(sliced);
  out.fut = out.sliced->Read();
  return out;
}

mdio::Result<std::string> FinishRead(Inflight& inf) {
  if (!inf.fut.has_value()) {
    return std::string();
  }
  MDIO_ASSIGN_OR_RETURN(auto data, inf.fut->result());
  std::string out;
  const Item& item = inf.item;
  const bool downsample = item.mx > 1 || item.my > 1 || item.mz > 1;
  if (!downsample) {
    CopyToFortranXyz(data, inf.six, inf.siy, inf.siz, item, inf.pad_x, inf.pad_y,
                     inf.pad_z, &out);
  } else if (data.dtype() == mdio::constants::kFloat32) {
    AverageToFortranXyz<float>(data, inf.six, inf.siy, inf.siz, item, inf.pad_x,
                               inf.pad_y, inf.pad_z, &out);
  } else if (data.dtype() == mdio::constants::kFloat64) {
    AverageToFortranXyz<double>(data, inf.six, inf.siy, inf.siz, item, inf.pad_x,
                                inf.pad_y, inf.pad_z, &out);
  } else if (data.dtype() == mdio::constants::kInt16) {
    AverageToFortranXyz<int16_t>(data, inf.six, inf.siy, inf.siz, item,
                                 inf.pad_x, inf.pad_y, inf.pad_z, &out);
  } else if (data.dtype() == mdio::constants::kInt32) {
    AverageToFortranXyz<int32_t>(data, inf.six, inf.siy, inf.siz, item,
                                 inf.pad_x, inf.pad_y, inf.pad_z, &out);
  } else if (data.dtype() == mdio::constants::kUint8) {
    AverageToFortranXyz<uint8_t>(data, inf.six, inf.siy, inf.siz, item,
                                 inf.pad_x, inf.pad_y, inf.pad_z, &out);
  } else {
    CopyToFortranXyz(data, inf.six, inf.siy, inf.siz, item, inf.pad_x, inf.pad_y,
                     inf.pad_z, &out);
  }
  return out;
}

class Server {
 public:
  explicit Server(tensorstore::Context ctx) : ctx_(std::move(ctx)) {}

  // Open one variable only. Dataset::Open reads every array in a LIBRARY and
  // hung the sidecar (hundreds of threads, minutes).
  mdio::Result<std::shared_ptr<mdio::Variable<>>> VariableFor(
      const std::string& dataset, const std::string& variable) {
    const std::string key = dataset + "\n" + variable;
    std::lock_guard<std::mutex> lock(mu_);
    auto it = variables_.find(key);
    if (it != variables_.end()) {
      return it->second;
    }
    MDIO_ASSIGN_OR_RETURN(auto loc, mdio::zarr::ResolveKvStoreLocation(dataset));
    nlohmann::json spec = mdio::zarr::BuildVariableSpec("zarr3", loc, variable);
    spec["recheck_cached_data"] = false;
    spec["recheck_cached_metadata"] = false;
    std::cerr << "mdio_load open zarr3 " << dataset << "/" << variable << "\n";
    auto opened =
        mdio::Variable<>::Open(spec, mdio::constants::kOpen, ctx_).result();
    if (!opened.ok()) {
      spec["driver"] = "zarr";
      std::cerr << "mdio_load fallback zarr2 " << dataset << "/" << variable
                << " err=" << opened.status() << "\n";
      opened =
          mdio::Variable<>::Open(spec, mdio::constants::kOpen, ctx_).result();
    }
    if (!opened.ok()) {
      return opened.status();
    }
    auto ptr = std::make_shared<mdio::Variable<>>(std::move(opened).value());
    const auto domain = ptr->dimensions();
    std::cerr << "mdio_load opened " << dataset << "/" << variable
              << " rank=" << domain.rank();
    for (mdio::DimensionIndex i = 0; i < domain.rank(); ++i) {
      std::cerr << " " << std::string(domain.labels()[i]) << "["
                << domain.origin()[i] << "+" << domain.shape()[i] << "]";
    }
    std::cerr << "\n";
    variables_[key] = ptr;
    return ptr;
  }

  std::pair<nlohmann::json, std::string> Handle(const Request& req) {
    auto var_res = VariableFor(req.dataset, req.variable);
    if (!var_res.ok()) {
      return {{{"ok", false},
               {"error", std::string(var_res.status().message())}},
              {}};
    }
    const mdio::Variable<> var = *var_res.value();
    std::vector<Inflight> inflight;
    inflight.reserve(req.items.size());
    for (const auto& item : req.items) {
      auto started = StartRead(var, req, item);
      if (!started.ok()) {
        return {{{"ok", false},
                 {"error", std::string(started.status().message())}},
                {}};
      }
      inflight.push_back(std::move(started).value());
    }
    std::vector<std::string> parts;
    parts.reserve(inflight.size());
    for (auto& inf : inflight) {
      auto finished = FinishRead(inf);
      if (!finished.ok()) {
        return {{{"ok", false},
                 {"error", std::string(finished.status().message())}},
                {}};
      }
      parts.push_back(std::move(finished).value());
    }
    nlohmann::json sizes = nlohmann::json::array();
    std::string payload;
    payload.reserve(parts.size() * 131072);
    for (const auto& p : parts) {
      sizes.push_back(p.size());
      payload.append(p);
    }
    return {{{"ok", true}, {"sizes", sizes}}, std::move(payload)};
  }

 private:
  tensorstore::Context ctx_;
  std::mutex mu_;
  std::unordered_map<std::string, std::shared_ptr<mdio::Variable<>>> variables_;
};

void ServeClient(int fd, Server* server) {
  while (true) {
    std::string raw;
    if (!ReadFrame(fd, &raw)) {
      break;
    }
    nlohmann::json reply;
    try {
      const auto j = nlohmann::json::parse(raw);
      if (j.value("op", "read") == "ping") {
        reply = {{"ok", true}};
        if (!WriteFrame(fd, reply.dump(), "")) {
          break;
        }
        continue;
      }
      auto req = ParseRequest(j);
      if (!req.ok()) {
        reply = {{"ok", false}, {"error", std::string(req.status().message())}};
        if (!WriteFrame(fd, reply.dump(), "")) {
          break;
        }
        continue;
      }
      auto handled = server->Handle(req.value());
      reply = std::move(handled.first);
      if (!reply.value("ok", false)) {
        if (!WriteFrame(fd, reply.dump(), "")) {
          break;
        }
      } else if (!WriteFrame(fd, reply.dump(), handled.second)) {
        break;
      }
    } catch (const std::exception& e) {
      reply = {{"ok", false}, {"error", e.what()}};
      if (!WriteFrame(fd, reply.dump(), "")) {
        break;
      }
    }
  }
  ::close(fd);
}

void PrintUsage() {
  std::cerr
      << "Usage: mdio_load [--port 0] [--cache-gb 8] [--threads 0] "
         "[--s3-concurrency 128]\n";
}

}  // namespace

int main(int argc, char** argv) {
  uint16_t port = 0;
  uint64_t cache_bytes = 8ull * 1024ull * 1024ull * 1024ull;
  int threads = 0;
  int s3_concurrency = 128;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return 0;
    }
    if (arg == "--port" && i + 1 < argc) {
      port = static_cast<uint16_t>(std::stoi(argv[++i]));
    } else if (arg.rfind("--port=", 0) == 0) {
      port = static_cast<uint16_t>(std::stoi(arg.substr(7)));
    } else if (arg == "--cache-gb" && i + 1 < argc) {
      cache_bytes = static_cast<uint64_t>(std::stoll(argv[++i])) * 1024ull *
                    1024ull * 1024ull;
    } else if (arg.rfind("--cache-gb=", 0) == 0) {
      cache_bytes = static_cast<uint64_t>(std::stoll(arg.substr(11))) * 1024ull *
                    1024ull * 1024ull;
    } else if (arg == "--threads" && i + 1 < argc) {
      threads = std::stoi(argv[++i]);
    } else if (arg.rfind("--threads=", 0) == 0) {
      threads = std::stoi(arg.substr(10));
    } else if (arg == "--s3-concurrency" && i + 1 < argc) {
      s3_concurrency = std::stoi(argv[++i]);
    } else if (arg.rfind("--s3-concurrency=", 0) == 0) {
      s3_concurrency = std::stoi(arg.substr(17));
    } else {
      std::cerr << "unknown arg " << arg << "\n";
      PrintUsage();
      return 2;
    }
  }

  const int copy_limit =
      threads > 0 ? threads
                  : static_cast<int>(std::thread::hardware_concurrency());
  nlohmann::json ctx_json = {
      {"cache_pool", {{"total_bytes_limit", cache_bytes}}},
      {"data_copy_concurrency", {{"limit", copy_limit > 0 ? copy_limit : 32}}},
      {"s3_request_concurrency",
       {{"limit", s3_concurrency > 0 ? s3_concurrency : 128}}}};
  auto spec = tensorstore::Context::Spec::FromJson(ctx_json);
  if (!spec.ok()) {
    ctx_json = {{"cache_pool", {{"total_bytes_limit", cache_bytes}}}};
    spec = tensorstore::Context::Spec::FromJson(ctx_json);
  }
  if (!spec.ok()) {
    std::cerr << "context spec failed: " << spec.status() << "\n";
    return 1;
  }
  tensorstore::Context ctx(*spec);

  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    std::cerr << "socket failed\n";
    return 1;
  }
  int yes = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    std::cerr << "bind failed\n";
    return 1;
  }
  socklen_t alen = sizeof(addr);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &alen) != 0) {
    std::cerr << "getsockname failed\n";
    return 1;
  }
  const uint16_t bound = ntohs(addr.sin_port);
  if (::listen(fd, 512) != 0) {
    std::cerr << "listen failed\n";
    return 1;
  }

  Server server(ctx);
  nlohmann::json ready = {{"ready", true}, {"port", bound}};
  std::cout << ready.dump() << std::endl;
  std::cerr << "mdio_load listening on 127.0.0.1:" << bound
            << " cache_bytes=" << cache_bytes << " copy_limit=" << copy_limit
            << " s3_concurrency=" << s3_concurrency << "\n";

  while (true) {
    const int cfd = ::accept(fd, nullptr, nullptr);
    if (cfd < 0) {
      continue;
    }
    int nodelay = 1;
    ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    std::thread(ServeClient, cfd, &server).detach();
  }
}
