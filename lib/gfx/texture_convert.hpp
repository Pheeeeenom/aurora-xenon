#pragma once

#include "texture.hpp"

namespace aurora::gfx {
inline bool is_pc_texture_format(u32 gxFormat) noexcept {
  return (gxFormat & _GX_TF_PC) != 0;
}

inline bool uses_direct_texture_upload(u32 gxFormat) noexcept {
  switch (gxFormat) {
  case GX_TF_R8_PC:
  case GX_TF_RG8_PC:
    return backend::limits().componentSwizzle;
  case GX_TF_RGBA8_PC:
    return true;
  case GX_TF_BC1_PC:
    return backend::limits().bcTextures;
  default:
    return false;
  }
}

inline TextureFormat upload_format(u32 gxFormat) noexcept {
  switch (gxFormat) {
  case GX_TF_R8_PC:
    return uses_direct_texture_upload(gxFormat) ? TextureFormat::R8Unorm : TextureFormat::RGBA8Unorm;
  case GX_TF_RG8_PC:
    return uses_direct_texture_upload(gxFormat) ? TextureFormat::RG8Unorm : TextureFormat::RGBA8Unorm;
  case GX_TF_C4:
  case GX_TF_C8:
  case GX_TF_C14X2:
    return TextureFormat::R16Sint;
  case GX_TF_BC1_PC:
    return uses_direct_texture_upload(gxFormat) ? TextureFormat::BC1RGBAUnorm : TextureFormat::RGBA8Unorm;
  default:
    return TextureFormat::RGBA8Unorm;
  }
}

// True if this CMPR texture uploads as BC1 instead of RGBA8.
bool cmpr_uses_bc1(uint32_t width, uint32_t height, uint32_t mips, ArrayRef<uint8_t> data) noexcept;

struct ConvertedTexture {
  TextureFormat format;
  uint32_t width;
  uint32_t height;
  uint32_t mips = 1;
  ByteBuffer data;
  bool hasArbitraryMips = false;
};

// Returns converted bytes when Aurora must transform the source layout before upload.
// Empty data means callers should upload the original bytes directly using upload_format(format).
// hasArbitraryMips is only meaningful for decoded RGBA8 GC formats; PC formats skip that check.
ConvertedTexture convert_texture(u32 format, uint32_t width, uint32_t height, uint32_t mips, ArrayRef<uint8_t> data);
ConvertedTexture convert_texture_palette(u32 textureFormat, uint32_t width, uint32_t height, uint32_t mips,
                                         ArrayRef<uint8_t> textureData, GXTlutFmt tlutFormat, uint16_t tlutEntries,
                                         ArrayRef<uint8_t> tlutData);
ConvertedTexture convert_tlut(u32 format, uint32_t width, ArrayRef<uint8_t> data);
GXTexFmt tlut_texture_format(GXTlutFmt format) noexcept;
// The base level as RGBA8, whatever it uploads as, for texture dumps.
ByteBuffer decode_rgba8(u32 format, uint32_t width, uint32_t height, ArrayRef<uint8_t> data, GXTlutFmt tlutFormat,
                        uint16_t tlutEntries, ArrayRef<uint8_t> tlutData);
} // namespace aurora::gfx
