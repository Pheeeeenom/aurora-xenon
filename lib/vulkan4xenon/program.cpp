// Programs: the pack's SPIR-V, else read from the disk, else compiled here by glslang and kept (xsdk shader-compiling).
#include "backend.hpp"

#include "../io.hpp"
#include "glsl.hpp"

#include <aurora/glsl_pack.h>
#include <vulkan4xenon.h>

#include <SPIRV/GlslangToSpv.h>
#include <fmt/format.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <malloc.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Without a pack linked in, none: every program comes from the disk or glslang.
extern "C" __attribute__((weak)) const AuroraGlslPack aurora_glsl_pack{};

namespace aurora::gfx::vulkan4xenon {
namespace {
constexpr uint32_t SpirvMagic = 0x07230203;
constexpr size_t SpirvHeaderWords = 5;

// Each of these is used under pipeline.cpp's g_pipelineMutex, one program at a time.

// The programs read or compiled this run, by hash; a node each, so a program's SPIR-V stays put as the map grows.
std::unordered_map<uint64_t, std::vector<uint32_t>> g_programs;
// Programs glslang refused, logged once.
std::unordered_set<uint64_t> g_refused;
// Where the SPIR-V compiled is kept; empty keeps none.
std::string g_folder;
bool g_glslangStarted = false;
uint32_t g_compiled = 0;
uint32_t g_read = 0;

Program from_pack(uint64_t hash) noexcept {
  const std::span programs{aurora_glsl_pack.programs, aurora_glsl_pack.count};
  const auto found = std::ranges::lower_bound(programs, hash, {}, &AuroraGlslProgram::hash);
  if (found != programs.end() && found->hash == hash) {
    return {found->spirv, found->bytes / sizeof(uint32_t)};
  }
  return {};
}

// A program kept by an earlier run: false when there's none, or the file doesn't hold SPIR-V.
bool read_kept(const std::string& path, std::vector<uint32_t>& spirv) noexcept {
  std::error_code error;
  const auto bytes = std::filesystem::file_size(io::fs_path_from_string(path), error);
  if (error) {
    return false;
  }
  if (bytes % sizeof(uint32_t) != 0 || bytes < SpirvHeaderWords * sizeof(uint32_t)) {
    Log.error("{} isn't SPIR-V: {} bytes", path, bytes);
    return false;
  }
  FILE* const file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    Log.error("{} doesn't open", path);
    return false;
  }
  spirv.resize(bytes / sizeof(uint32_t));
  const bool read = std::fread(spirv.data(), sizeof(uint32_t), spirv.size(), file) == spirv.size();
  std::fclose(file);
  if (!read || spirv[0] != SpirvMagic) {
    Log.error("{} isn't SPIR-V", path);
    return false;
  }
  return true;
}

// Written beside its name and renamed into place, so a crash leaves no file half written.
void keep(const std::string& path, const std::vector<uint32_t>& spirv) noexcept {
  const std::string written = path + ".new";
  FILE* const file = std::fopen(written.c_str(), "wb");
  bool kept = file != nullptr && std::fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), file) == spirv.size();
  if (file != nullptr && std::fclose(file) != 0) {
    kept = false;
  }
  kept = kept && std::rename(written.c_str(), path.c_str()) == 0;
  if (!kept) {
    Log.error("{} not kept", path);
    if (file != nullptr) {
      std::remove(written.c_str());
    }
  }
}

// As `glslang -V` builds the pack, with the settings that gave the PC's SPIR-V (glslang-xenon:xsdk/glslang_probe.cpp).
bool compile(const std::string& source, bool fragment, const std::string& name, std::vector<uint32_t>& spirv) noexcept {
  if (!g_glslangStarted) {
    glslang::InitializeProcess();
    g_glslangStarted = true;
  }
  const EShLanguage stage = fragment ? EShLangFragment : EShLangVertex;
  const auto messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
  glslang::TShader shader{stage};
  const char* const text = source.c_str();
  shader.setStrings(&text, 1);
  shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
  shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
  shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
  if (!shader.parse(GetDefaultResources(), 100, false, messages)) {
    Log.error("glslang refused {}: {}", name, shader.getInfoLog());
    return false;
  }
  glslang::TProgram program;
  program.addShader(&shader);
  if (!program.link(messages)) {
    Log.error("glslang didn't link {}: {}", name, program.getInfoLog());
    return false;
  }
  glslang::SpvOptions options;
  options.disableOptimizer = false;
  std::vector<unsigned int> words;
  glslang::GlslangToSpv(*program.getIntermediate(stage), words, &options);
  spirv.assign(words.begin(), words.end());
  return true;
}
} // namespace

void initialize_programs() noexcept {
  Log.info("Shader pack: {} programs", aurora_glsl_pack.count);
  if (g_config.cachePath == nullptr) {
    Log.warn("No cache path: programs compiled here aren't kept");
    return;
  }
  // glslang's SPIR-V in a folder named after its version; vulkan4xenon keeps its shaders beside it.
  const auto shaders = io::fs_path_from_string(g_config.cachePath) / "shaders";
  const auto version = glslang::GetVersion();
  const auto folder = shaders / fmt::format("glslang-{}.{}.{}{}{}", version.major, version.minor, version.patch,
                                            *version.flavor != '\0' ? "-" : "", version.flavor);
  std::error_code error;
  std::filesystem::create_directories(folder, error);
  if (error) {
    Log.error("{} wasn't made: {}; programs compiled here aren't kept", io::fs_path_to_string(folder), error.message());
    return;
  }
  g_folder = io::fs_path_to_string(folder);
  v4x_set_shader_cache(io::fs_path_to_string(shaders).c_str());
  Log.info("Programs the pack doesn't hold are kept in {}", g_folder);
}

void shutdown_programs() noexcept {
  const V4xShaderCounts layer = v4x_shader_counts();
  Log.info("Programs this run: {} compiled and {} read from the disk; vulkan4xenon's shaders: {} compiled and {} read",
           g_compiled, g_read, layer.compiled, layer.read);
  g_programs.clear();
  g_refused.clear();
}

Program find_program(const std::string& source, bool fragment) noexcept {
  const uint64_t hash = glsl::program_hash(source, fragment);
  if (const Program packed = from_pack(hash); !packed.empty()) {
    return packed;
  }
  if (const auto found = g_programs.find(hash); found != g_programs.end()) {
    return found->second;
  }
  if (g_refused.contains(hash)) {
    return {};
  }
  const std::string name = fmt::format("{:016x}.{}", hash, fragment ? "frag" : "vert");
  const std::string path = g_folder.empty() ? std::string{} : fmt::format("{}/{}.spv", g_folder, name);
  std::vector<uint32_t> spirv;
  if (!path.empty() && read_kept(path, spirv)) {
    ++g_read;
    Log.info("Read {} from the disk; {} compiled and {} read this run", name, g_compiled, g_read);
  } else {
    const auto start = std::chrono::steady_clock::now();
    if (!compile(source, fragment, name, spirv)) {
      g_refused.insert(hash);
      return {};
    }
    const auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    ++g_compiled;
    Log.info("Compiled {} in {:.1f} ms, with {} KB of heap in use; {} compiled and {} read this run", name,
             static_cast<double>(us) / 1000.0, mallinfo().uordblks / 1024, g_compiled, g_read);
    if (!path.empty()) {
      keep(path, spirv);
    }
  }
  return g_programs.emplace(hash, std::move(spirv)).first->second;
}
} // namespace aurora::gfx::vulkan4xenon
