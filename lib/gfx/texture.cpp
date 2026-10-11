#include "resources.hpp"
#include "recording.hpp"

#include "../internal.hpp"
#include "aurora/aurora.h"
#include "texture.hpp"
#include "texture_convert.hpp"
#include "../gx/gx_fmt.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include <fmt/format.h>
#include <tracy/Tracy.hpp>

namespace aurora::gfx {
namespace {
constexpr Module Log{"aurora::gfx"};

TextureExtent physical_size(TextureExtent size, TextureFormatInfo info) {
  const uint32_t width = ((size.width + info.blockWidth - 1) / info.blockWidth) * info.blockWidth;
  const uint32_t height = ((size.height + info.blockHeight - 1) / info.blockHeight) * info.blockHeight;
  return {.width = width, .height = height};
}

std::pair<SamplerFilter, SamplerMipFilter> filter_mode(GXTexFilter filter) {
  switch (filter) {
    DEFAULT_FATAL("invalid filter mode {}", static_cast<int>(filter));
  case GX_NEAR:
    return {SamplerFilter::Nearest, SamplerMipFilter::None};
  case GX_LINEAR:
    return {SamplerFilter::Linear, SamplerMipFilter::None};
  case GX_NEAR_MIP_NEAR:
    return {SamplerFilter::Nearest, SamplerMipFilter::Nearest};
  case GX_LIN_MIP_NEAR:
    return {SamplerFilter::Linear, SamplerMipFilter::Nearest};
  case GX_NEAR_MIP_LIN:
    return {SamplerFilter::Nearest, SamplerMipFilter::Linear};
  case GX_LIN_MIP_LIN:
    return {SamplerFilter::Linear, SamplerMipFilter::Linear};
  }
}

u16 anisotropy(GXAnisotropy aniso) {
  switch (aniso) {
    DEFAULT_FATAL("invalid aniso {}", static_cast<int>(aniso));
  case GX_ANISO_1:
  case GX_MAX_ANISOTROPY:
    return 1;
  case GX_ANISO_2:
    return std::max<u16>(g_config.maxTextureAnisotropy / 2, 1);
  case GX_ANISO_4:
    return std::max<u16>(g_config.maxTextureAnisotropy, 1);
  }
}
} // namespace

// The upload half of new_static_texture_2d, shared with new_static_texture_2d_converted.
static void queue_static_texture_upload(TextureRef& ref, uint32_t mips, ArrayRef<uint8_t> data,
                                        const char* label) noexcept {
  uint32_t offset = 0;
  for (uint32_t mip = 0; mip < mips; ++mip) {
    const TextureExtent mipSize{
        .width = std::max(ref.size.width >> mip, 1u),
        .height = std::max(ref.size.height >> mip, 1u),
    };
    const auto info = format_info(ref.format);
    const auto physicalSize = physical_size(mipSize, info);
    const uint32_t widthBlocks = physicalSize.width / info.blockWidth;
    const uint32_t heightBlocks = physicalSize.height / info.blockHeight;
    const uint32_t bytesPerRow = widthBlocks * info.blockSize;
    const uint32_t dataSize = bytesPerRow * heightBlocks;
    CHECK(offset + dataSize <= data.size(), "new_static_texture_2d[{}]: expected at least {} bytes, got {}", label,
          offset + dataSize, data.size());
    queue_texture_upload_data(data.data() + offset,
                              {
                                  .texture = ref.gpu,
                                  .mipLevel = mip,
                                  .bytesPerRow = bytesPerRow,
                                  .rowsPerImage = heightBlocks,
                                  .width = physicalSize.width,
                                  .height = physicalSize.height,
                              },
                              info.blockSize);
    offset += dataSize;
  }
  if (data.size() != UINT32_MAX && offset < data.size()) {
    Log.warn("new_static_texture_2d[{}]: texture used {} bytes, but given {} bytes", label, offset, data.size());
  }
}

TextureHandle new_static_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 format, ArrayRef<uint8_t> data,
                                    bool tlut, const char* label) noexcept {
  ZoneScoped;

  // Converted before the texture is made, since CMPR's content decides its format.
  ConvertedTexture converted{};
  if (format != InvalidTextureFormat) {
    if (tlut) {
      CHECK(height == 1, "new_static_texture_2d[{}]: expected tlut height 1, got {}", label, height);
      CHECK(mips == 1, "new_static_texture_2d[{}]: expected tlut mipCount 1, got {}", label, mips);
      converted = convert_tlut(format, width, data);
    } else {
      converted = convert_texture(format, width, height, mips, data);
    }
    if (!converted.data.empty()) {
      data = converted.data;
    }
  }

  const auto uploadFormat = converted.format != TextureFormat::Undefined ? converted.format : upload_format(format);
  auto handle = new_texture(width, height, mips, uploadFormat, format, backend::TextureUsage::Sampled, label);
  auto& ref = *handle;
  ref.hasArbitraryMips = converted.hasArbitraryMips;
  queue_static_texture_upload(ref, mips, data, label);
  return handle;
}

TextureHandle new_static_texture_2d_converted(uint32_t width, uint32_t height, uint32_t mips, u32 format,
                                              TextureFormat uploadFormat, ArrayRef<uint8_t> converted,
                                              bool hasArbitraryMips, const char* label) noexcept {
  ZoneScoped;
  auto handle = new_texture(width, height, mips, uploadFormat, format, backend::TextureUsage::Sampled, label);
  auto& ref = *handle;
  ref.hasArbitraryMips = hasArbitraryMips;
  queue_static_texture_upload(ref, mips, converted, label);
  return handle;
}

TextureHandle new_dynamic_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                     const char* label) noexcept {
  ZoneScopedS(3);
  return new_texture(width, height, mips, upload_format(gxFormat), gxFormat, backend::TextureUsage::Sampled, label);
}

TextureHandle new_render_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  ZoneScoped;
  const auto format = backend::efb_layout().colorAttachments[SceneColorAttachmentIndex].format;
  return new_texture(width, height, 1, format, gxFormat, backend::TextureUsage::Copy, label);
}

TextureHandle new_conv_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  ZoneScoped;
  return new_texture(width, height, 1, upload_format(gxFormat), gxFormat, backend::TextureUsage::Conv, label);
}

TextureHandle new_texture(uint32_t width, uint32_t height, uint32_t mips, TextureFormat format, u32 gxFormat,
                          backend::TextureUsage usage, const char* label) noexcept {
  auto gpu = backend::create_texture({
      .width = width,
      .height = height,
      .mips = mips,
      .format = format,
      .gxFormat = gxFormat,
      .usage = usage,
      .label = label,
  });
  return std::make_shared<TextureRef>(std::move(gpu), TextureExtent{width, height}, format, mips, gxFormat);
}

bool write_texture_levels(TextureRef& ref, TextureFormatInfo info, ArrayRef<uint8_t> data) noexcept {
  uint32_t offset = 0;
  for (uint32_t mip = 0; mip < ref.mipCount; ++mip) {
    const TextureExtent mipSize{
        .width = std::max(ref.size.width >> mip, 1u),
        .height = std::max(ref.size.height >> mip, 1u),
    };
    const auto physicalSize = physical_size(mipSize, info);
    const uint32_t widthBlocks = physicalSize.width / info.blockWidth;
    const uint32_t heightBlocks = physicalSize.height / info.blockHeight;
    const uint32_t bytesPerRow = widthBlocks * info.blockSize;
    const uint32_t dataSize = bytesPerRow * heightBlocks;
    CHECK(offset + dataSize <= data.size(), "write_texture_levels: expected at least {} bytes, got {}",
          offset + dataSize, data.size());
    if (!queue_texture_upload_data(data.data() + offset,
                                   {
                                       .texture = ref.gpu,
                                       .mipLevel = mip,
                                       .bytesPerRow = bytesPerRow,
                                       .rowsPerImage = heightBlocks,
                                       .width = physicalSize.width,
                                       .height = physicalSize.height,
                                   },
                                   info.blockSize)) {
      return false;
    }
    offset += dataSize;
  }
  if (data.size() != UINT32_MAX && offset < data.size()) {
    Log.warn("write_texture_levels: texture used {} bytes, but given {} bytes", offset, data.size());
  }
  return true;
}

SamplerConfig TextureBind::sampler_config() const noexcept {
  auto [minFilter, mipFilter] = filter_mode(texObj.min_filter());
  auto [magFilter, _] = filter_mode(texObj.mag_filter());
  const bool mipsEnabled = mipFilter != SamplerMipFilter::None;
  float minLod = texObj.min_lod();
  float maxLod = texObj.max_lod();
  u16 maxAnisotropy = anisotropy(texObj.max_aniso());
  if (ref && ref->isReplacement) {
    minLod = 0.f;
    maxLod = 1000.f;
    if (!mipsEnabled) {
      mipFilter = SamplerMipFilter::Nearest;
    }
  } else if (mipFilter == SamplerMipFilter::None) {
    minLod = 0.f;
    maxLod = 0.f;
  }
  if ((ref && ref->hasArbitraryMips) || !mipsEnabled) {
    maxAnisotropy = 1;
  } else if (maxAnisotropy > 1) {
    magFilter = SamplerFilter::Linear;
    minFilter = SamplerFilter::Linear;
    mipFilter = SamplerMipFilter::Linear;
  }
  return {
      .wrapS = texObj.wrap_s(),
      .wrapT = texObj.wrap_t(),
      .magFilter = magFilter,
      .minFilter = minFilter,
      .mipFilter = mipFilter,
      .minLod = minLod,
      .maxLod = maxLod,
      .maxAnisotropy = maxAnisotropy,
  };
}

// Texture addresses and sampler fields: equal exactly when the WebGPU bind group descriptors were, as each texture has
// one view and each sampler descriptor one cached sampler. The fields are packed one after another and hashed at once,
// which gives the key that streaming them through 72 Hasher::update calls gave (XXH3's streaming and one-shot results
// are the same), at a fraction of the cost.
BindGroupRef TextureSet::hash() const noexcept {
  constexpr size_t MapBytes = sizeof(uintptr_t) + 2 * sizeof(uint32_t) + sizeof(SamplerFilter) * 2 +
                              sizeof(SamplerMipFilter) + 2 * sizeof(uint32_t) + sizeof(uint16_t);
  std::array<uint8_t, MapBytes * GX_MAX_TEXMAP> bytes;
  uint8_t* out = bytes.data();
  const auto put = [&out]<typename T>(const T& value) {
    std::memcpy(out, &value, sizeof(T));
    out += sizeof(T);
  };
  for (size_t i = 0; i < textures.size(); ++i) {
    const auto& sampler = samplers[i];
    put(reinterpret_cast<uintptr_t>(textures[i].get()));
    put(static_cast<uint32_t>(sampler.wrapS));
    put(static_cast<uint32_t>(sampler.wrapT));
    put(sampler.magFilter);
    put(sampler.minFilter);
    put(sampler.mipFilter);
    put(std::bit_cast<uint32_t>(sampler.minLod));
    put(std::bit_cast<uint32_t>(sampler.maxLod));
    put(sampler.maxAnisotropy);
  }
  return static_cast<BindGroupRef>(XXH3_64bits(bytes.data(), bytes.size()));
}
} // namespace aurora::gfx
