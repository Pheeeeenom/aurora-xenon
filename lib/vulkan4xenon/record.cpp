// The GLSL each GX shader config needs, written out as a game asks for it, for the offline step that compiles it for
// the Xbox 360 (vulkan4xenon's ROADMAP.md, milestone 10). A backend that records needs no compiler of its own.
#include "glsl.hpp"

#include "../internal.hpp"

#include <fmt/format.h>

#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_set>

namespace aurora::gfx::vulkan4xenon::glsl {
namespace {
constexpr Module Log{"aurora::gfx::vulkan4xenon::record"};

std::mutex s_lock;
std::unordered_set<uint64_t> s_written;
bool s_madeDir = false;

void write(const std::filesystem::path& dir, const std::string& source, bool fragment) {
  const uint64_t hash = program_hash(source, fragment);
  if (!s_written.insert(hash).second) {
    return;
  }
  const auto name = fmt::format("{:016x}.{}", hash, fragment ? "frag" : "vert");
  std::ofstream file(dir / name, std::ios::binary);
  file << source;
  if (!file) {
    Log.error("couldn't write {}", (dir / name).string());
    return;
  }
  Log.info("recorded {}", name);
}
} // namespace

void record(const gx::ShaderConfig& config, const char* dir) {
  const Programs programs = build_programs(config);
  std::lock_guard lock(s_lock);
  const std::filesystem::path path(dir);
  if (!s_madeDir) {
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) {
      Log.error("couldn't make {}: {}", path.string(), error.message());
    }
    s_madeDir = true;
    for (const auto& [source, fragment] : fixed_programs()) {
      write(path, source, fragment);
    }
  }
  if (!programs.vertex.empty()) {
    write(path, programs.vertex, false);
    write(path, programs.fragment, true);
  }
}
} // namespace aurora::gfx::vulkan4xenon::glsl
