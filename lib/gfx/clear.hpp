#pragma once

#include <aurora/gfx.hpp>

#include "texture.hpp"
#include "types.hpp"

#include <array>

namespace aurora::gfx::clear {
struct DrawData {
  PipelineRef pipeline;
  std::array<float, 4> color;
  float depth = 0.f;
  ClipRect rect{};
};

constexpr uint32_t ClearPipelineConfigVersion = 5;
struct PipelineConfig {
  uint32_t version = ClearPipelineConfigVersion;
  bool clearColor = true;
  bool clearAlpha = true;
  bool clearDepth = true;
  uint8_t _pad = 0;
};
static_assert(std::has_unique_object_representations_v<PipelineConfig>);
} // namespace aurora::gfx::clear
