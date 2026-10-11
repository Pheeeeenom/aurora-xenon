#pragma once

#include "types.hpp"

#include <dolphin/gx/GXEnum.h>

#include <array>

namespace aurora::gfx::tex_copy_conv {

// The EFB copy's source rectangle and alpha, which the recorder pushes for the copy.
struct alignas(16) Uniforms {
  Vec2<float> offset;
  Vec2<float> scale{1.f, 1.f};
  uint32_t opaqueAlpha = 0;
  std::array<uint32_t, 3> _pad{};
};
static_assert(sizeof(Uniforms) == 32);

} // namespace aurora::gfx::tex_copy_conv
