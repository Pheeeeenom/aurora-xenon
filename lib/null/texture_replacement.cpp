// Texture packs and dumps load only under the Dawn backend; here they report themselves unsupported.
#include "../gfx/texture_replacement.hpp"
#include "../internal.hpp"

#include <aurora/replacement.h>

#include <mutex>

namespace {
constexpr aurora::Module Log{"aurora::gfx::texture_replacement"};

void report_unsupported() noexcept {
  static std::once_flag once;
  std::call_once(once, [] { Log.warn("Texture packs and dumps are supported only by the Dawn backend"); });
}
} // namespace

extern "C" uint32_t aurora_replacement_load(const char* root, int32_t priority, int one_pack) {
  (void)root;
  (void)priority;
  (void)one_pack;
  report_unsupported();
  return 0;
}

extern "C" void aurora_replacement_clear() {}

extern "C" void aurora_replacement_set_cache_budget(uint64_t bytes) { (void)bytes; }

extern "C" void aurora_replacement_prefetch(const GXTexObj* obj, const GXTlutObj* tlut) {
  (void)obj;
  (void)tlut;
}

extern "C" void aurora_replacement_set_wait(int wait) { (void)wait; }

extern "C" void aurora_replacement_set_strict_budget(int strict) { (void)strict; }

extern "C" int aurora_replacement_dump(const GXTexObj* obj, const GXTlutObj* tlut, const char* dir) {
  (void)obj;
  (void)tlut;
  (void)dir;
  report_unsupported();
  return -1;
}

namespace aurora::gfx::texture_replacement {
void shutdown() noexcept {}

StreamingStats process_streaming() noexcept { return {}; }

std::optional<ReplacementResult> find_pointer_replacement(const GXTexObj_& obj) noexcept {
  (void)obj;
  return std::nullopt;
}

std::optional<ReplacementResult> find_source_replacement(const GXTexObj_& obj,
                                                         const texture::TextureSourceKey& sourceKey) noexcept {
  (void)obj;
  (void)sourceKey;
  return std::nullopt;
}

bool should_build_source_key() noexcept { return false; }

std::string build_texture_replacement_name(const texture::TextureSourceKey& sourceKey) noexcept {
  (void)sourceKey;
  return {};
}
} // namespace aurora::gfx::texture_replacement

// The C++ calls a game makes: nothing registers, so no texture is ever replaced.
namespace aurora::texture {
std::optional<TextureSourceKey> parse_replacement_filename(std::string_view filename) noexcept {
  (void)filename;
  report_unsupported();
  return std::nullopt;
}

ReplacementRegistration register_virtual_replacement(std::string_view path, VirtualFileSource source,
                                                     ReplacementOptions options) {
  (void)path;
  (void)source;
  (void)options;
  report_unsupported();
  return {};
}

ReplacementRegistration register_replacement(ReplacementKey key, RawTextureReplacement replacement,
                                             ReplacementOptions options) {
  (void)key;
  (void)replacement;
  (void)options;
  report_unsupported();
  return {};
}

void unregister_replacement(const ReplacementRegistration& registration) { (void)registration; }

void unregister_replacements(const ReplacementGroup& group) { (void)group; }

ReplacementGroup load_replacement_directory(const std::filesystem::path& root, ReplacementOptions options) {
  (void)root;
  (void)options;
  report_unsupported();
  return {};
}

bool has_replacement(const GXTexObj* obj, const GXTlutObj* tlut) {
  (void)obj;
  (void)tlut;
  return false;
}
} // namespace aurora::texture
