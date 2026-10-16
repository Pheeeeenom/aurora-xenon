#pragma once

#include "gx.hpp"

namespace aurora::gx {
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept;
// Pushes each piece in pieces (UniformPiece bits) as the state has it now, and notes where it went.
void push_uniforms(const ShaderInfo& info, u8 pieces, std::array<u32, UniformPieceCount>& offsets) noexcept;
u8 color_channel(GXChannelID id) noexcept;
}; // namespace aurora::gx
