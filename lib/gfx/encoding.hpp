#pragma once

#include "frame_packet.hpp"

#include <webgpu/webgpu_cpp.h>

namespace aurora::gfx {

bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass);
void bind_pipeline(const wgpu::RenderPipeline& pipeline, const wgpu::RenderPassEncoder& pass);

} // namespace aurora::gfx
