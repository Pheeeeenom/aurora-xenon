#pragma once
// A 32-bit texel's index in Xenos's tiled layout, as xsdk's tests read resolves back (xsdk:tests/tiled_index.h). It
// comes from libxenon:drivers/console/console.c:60-61, which carries this notice:
//
// Copyright 2009  Georg Lukas <georg@op-co.de>
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.

#include <cstdint>

namespace aurora::gfx::vulkan4xenon {
// The texel at (x, y) of a texture whose rows are pitch texels, a multiple of 32: its tiles are 32 by 32 texels.
constexpr uint32_t tiled_index(uint32_t x, uint32_t y, uint32_t pitch) noexcept {
  return ((y >> 5) * 32 * pitch + ((x >> 5) << 10) + (x & 3) + ((y & 1) << 2) + (((x & 31) >> 2) << 3) +
          (((y & 31) >> 1) << 6)) ^
         ((y & 8) << 2);
}
} // namespace aurora::gfx::vulkan4xenon
