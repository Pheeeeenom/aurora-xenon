#pragma once

#include "backend.hpp"
#include "types.hpp"

namespace aurora::gfx::clear {
struct PipelineConfig;
} // namespace aurora::gfx::clear

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::rmlui {
struct PipelineConfig;
} // namespace aurora::rmlui

namespace aurora::gfx {

enum class ShaderType : uint8_t {
  Clear = 0,
  GX = 1,
  Rml = 2,
};

void initialize_pipeline_cache();
void shutdown_pipeline_cache();
void begin_pipeline_frame();
void end_pipeline_frame();
void rebuild_pipeline_cache();

PipelineRef find_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout);
PipelineRef find_pipeline(const clear::PipelineConfig& config, const RenderTargetLayout& layout);
PipelineRef find_pipeline(const rmlui::PipelineConfig& config);

// False until the pipeline exists.
bool get_pipeline(PipelineRef ref, CompiledPipeline& pipeline);
void get_pipeline_counts(uint32_t& queued, uint32_t& created);

} // namespace aurora::gfx
