#pragma once
// Texture pack data in any format Dawn samples; texture packs are Dawn-only.

#include "texture.hpp"

#include <webgpu/webgpu_cpp.h>

namespace aurora::gfx {
TextureFormatInfo format_info(wgpu::TextureFormat format) noexcept;
uint64_t calc_texture_size(wgpu::TextureFormat format, uint32_t width, uint32_t height, uint32_t mips) noexcept;
bool is_block_aligned(wgpu::TextureFormat format, uint32_t width, uint32_t height) noexcept;

struct ReplacementTexture {
  wgpu::TextureFormat format;
  uint32_t width;
  uint32_t height;
  uint32_t mips = 1;
  ByteBuffer data;
  bool hasArbitraryMips = false;
};
} // namespace aurora::gfx
