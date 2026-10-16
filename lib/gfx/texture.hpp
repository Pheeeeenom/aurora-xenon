#pragma once
#include <dolphin/gx.h>

#include <utility>

#include "backend.hpp"
#include "types.hpp"

namespace aurora::gfx {
// One level's copy into a backend texture, from the frame's upload pool or from overflow.
struct TextureUpload {
  std::shared_ptr<backend::Texture> texture;
  uint32_t mipLevel = 0;
  uint32_t offset = 0;
  uint32_t bytesPerRow = 0;
  uint32_t rowsPerImage = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  // Array layers, each rowsPerImage rows; only the WebGPU extension's uploads have more than one.
  uint32_t layers = 1;
  // The rows, staged by the backend, when the pool had no room.
  std::shared_ptr<backend::UploadBuffer> overflow;
};
void queue_texture_upload(TextureUpload upload);
// False when the backend had no memory to stage it; nothing is queued then. The rows go at an offset that is a multiple
// of blockSize, the format's bytes a block, as Vulkan's buffer-to-image copies ask.
bool queue_texture_upload_data(const uint8_t* data, TextureUpload upload, uint32_t blockSize);

struct TextureFormatInfo {
  uint8_t blockWidth;
  uint8_t blockHeight;
  uint8_t blockSize;
  bool compressed;
};
TextureFormatInfo format_info(TextureFormat format) noexcept;
uint64_t calc_texture_size(TextureFormat format, uint32_t width, uint32_t height, uint32_t mips) noexcept;

constexpr u32 InvalidTextureFormat = -1;
struct TextureExtent {
  uint32_t width;
  uint32_t height;
};
struct TextureRef {
  std::shared_ptr<backend::Texture> gpu;
  TextureExtent size;
  // Undefined for a texture pack's format, which only the backend knows.
  TextureFormat format;
  uint32_t mipCount;
  u32 gxFormat;
  bool hasArbitraryMips = false;
  bool isReplacement = false;

  TextureRef(std::shared_ptr<backend::Texture> gpu, TextureExtent size, TextureFormat format, uint32_t mipCount,
             u32 gxFormat)
  : gpu(std::move(gpu)), size(size), format(format), mipCount(mipCount), gxFormat(gxFormat) {}
};

TextureHandle new_static_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                    ArrayRef<uint8_t> data, bool tlut, const char* label) noexcept;
// new_static_texture_2d for data convert_texture already produced from this texture.
TextureHandle new_static_texture_2d_converted(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                              TextureFormat format, ArrayRef<uint8_t> converted, bool hasArbitraryMips,
                                              const char* label) noexcept;
TextureHandle new_dynamic_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                     const char* label) noexcept;
TextureHandle new_render_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept;
TextureHandle new_conv_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept;
// A texture created as the backend is asked, with nothing uploaded.
TextureHandle new_texture(uint32_t width, uint32_t height, uint32_t mips, TextureFormat format, u32 gxFormat,
                          backend::TextureUsage usage, const char* label) noexcept;
// Uploads data already in the texture's format; false when a level could not be staged.
bool write_texture_levels(TextureRef& ref, TextureFormatInfo info, ArrayRef<uint8_t> data) noexcept;
}; // namespace aurora::gfx

struct GXTexObj_ {
  u32 mode0 = 0;
  u32 mode1 = 0;
  u32 image0 = UINT32_MAX;
  u32 image3 = 0;
  const void* userData = nullptr;
  const void* data = nullptr;
  u32 mWidth = 0;
  u32 mHeight = 0;
  u32 mFormat = aurora::gfx::InvalidTextureFormat;
  GXTlut tlut = GX_TLUT0;
  u32 texObjId = 0;
  u32 texDataVersion = 0;
  u8 flags = 0;

  static constexpr u32 get_bits(u32 reg, u32 size, u32 shift) noexcept { return (reg >> shift) & ((1u << size) - 1); }

  u32 width() const noexcept { return mWidth != 0 ? mWidth : image0 == UINT32_MAX ? 0 : get_bits(image0, 10, 0) + 1; }
  u32 height() const noexcept {
    return mHeight != 0 ? mHeight : image0 == UINT32_MAX ? 0 : get_bits(image0, 10, 10) + 1;
  }
  u32 raw_format() const noexcept { return get_bits(image0, 4, 20); }
  u32 format() const noexcept { return mFormat != aurora::gfx::InvalidTextureFormat ? mFormat : raw_format(); }
  GXTexWrapMode wrap_s() const noexcept { return static_cast<GXTexWrapMode>(get_bits(mode0, 2, 0)); }
  GXTexWrapMode wrap_t() const noexcept { return static_cast<GXTexWrapMode>(get_bits(mode0, 2, 2)); }
  GXTexFilter min_filter() const noexcept {
    constexpr GXTexFilter kHwToGxFilter[8] = {
        GX_NEAR, GX_NEAR_MIP_NEAR, GX_LIN_MIP_NEAR, GX_NEAR, GX_LINEAR, GX_NEAR_MIP_LIN, GX_LIN_MIP_LIN, GX_NEAR,
    };
    return kHwToGxFilter[get_bits(mode0, 3, 5)];
  }
  GXTexFilter mag_filter() const noexcept { return get_bits(mode0, 1, 4) != 0 ? GX_LINEAR : GX_NEAR; }
  GXBool has_mips() const noexcept { return (flags & 1u) != 0 ? GX_TRUE : GX_FALSE; }
  u32 mip_count() const noexcept { return has_mips() ? std::max<u32>(static_cast<u32>(max_lod()) + 1, 1u) : 1; }
  GXBool do_edge_lod() const noexcept { return get_bits(mode0, 1, 8) == 0 ? GX_TRUE : GX_FALSE; }
  float lod_bias() const noexcept { return static_cast<float>(static_cast<int8_t>(get_bits(mode0, 8, 9))) / 32.0f; }
  GXAnisotropy max_aniso() const noexcept { return static_cast<GXAnisotropy>(get_bits(mode0, 2, 19)); }
  GXBool bias_clamp() const noexcept { return get_bits(mode0, 1, 21) != 0 ? GX_TRUE : GX_FALSE; }
  float min_lod() const noexcept { return static_cast<float>(get_bits(mode1, 8, 0)) / 16.0f; }
  float max_lod() const noexcept { return static_cast<float>(get_bits(mode1, 8, 8)) / 16.0f; }

  // Custom flag for texture caching
  bool no_cache() const noexcept { return (flags & 0x80) != 0; }
  void set_no_cache(bool value) noexcept { flags = value ? flags | 0x80 : flags & ~0x80; }

  // Hacky workaround for an instances where incremental IDs are used for GXCopyTex, but the copy tex was invalidated
  // and the texture reference is still present.
  bool has_data() const noexcept { return reinterpret_cast<uintptr_t>(data) >= 0x10000; }
};
static_assert(sizeof(GXTexObj_) <= sizeof(GXTexObj), "GXTexObj too small!");
struct GXTlutObj_ {
  u32 tlut = 0;
  u32 loadTlut0 = 0;
  u16 numEntries = 0;
  const void* data = nullptr;
  GXTlutFmt format = GX_TL_IA8;
  u32 tlutObjId = 0;
  u32 tlutDataVersion = 0;
  u8 flags = 0;

  // Custom flag for texture caching
  bool no_cache() const noexcept { return (flags & 0x80) != 0; }
  void set_no_cache(bool value) noexcept { flags = value ? flags | 0x80 : flags & ~0x80; }
};
static_assert(sizeof(GXTlutObj_) <= sizeof(GXTlutObj), "GXTlutObj too small!");

namespace aurora::gfx {
// A GX texture's sampler after Aurora's LOD and anisotropy rules, for the backend to create.
enum class SamplerFilter : uint8_t {
  Nearest,
  Linear,
};
enum class SamplerMipFilter : uint8_t {
  None,
  Nearest,
  Linear,
};
struct SamplerConfig {
  GXTexWrapMode wrapS = GX_CLAMP;
  GXTexWrapMode wrapT = GX_CLAMP;
  SamplerFilter magFilter = SamplerFilter::Nearest;
  SamplerFilter minFilter = SamplerFilter::Nearest;
  SamplerMipFilter mipFilter = SamplerMipFilter::None;
  float minLod = 0.f;
  float maxLod = 0.f;
  uint16_t maxAnisotropy = 1;
  // The texture's LOD bias, which a backend may apply in its sampler, as vulkan4xenon's does; WebGPU's programs add it
  // themselves (gx/shader.cpp).
  float lodBias = 0.f;
};

struct TextureBind {
  TextureHandle ref;
  GXTexObj_ texObj;
  uint64_t generation = 0;

  TextureBind() noexcept = default;
  TextureBind(const GXTexObj_& obj, TextureHandle handle, uint64_t bindGeneration = 0) noexcept
  : ref(std::move(handle)), texObj(obj), generation(bindGeneration) {}
  void reset() noexcept { ref.reset(); }
  [[nodiscard]] SamplerConfig sampler_config() const noexcept;
  operator bool() const noexcept { return ref.operator bool(); }
};

// What a draw samples; slots it doesn't sample are empty.
struct TextureSet {
  std::array<TextureHandle, GX_MAX_TEXMAP> textures;
  std::array<SamplerConfig, GX_MAX_TEXMAP> samplers;

  // Equal for equal sets, so the backend can cache what it makes of one. It hashes texture addresses, so a backend
  // that caches by it keeps the set's textures alive while cached.
  [[nodiscard]] BindGroupRef hash() const noexcept;
};
} // namespace aurora::gfx
