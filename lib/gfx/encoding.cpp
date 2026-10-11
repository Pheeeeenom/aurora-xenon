#include "encoding.hpp"

#include "frame.hpp"

#include "clear.hpp"
#include "depth_peek.hpp"
#include "pipeline_cache.hpp"
#include "tex_copy_conv.hpp"
#include "tex_palette_conv.hpp"
#include "../gx/gx.hpp"
#include "../gx/pipeline.hpp"
#ifdef AURORA_ENABLE_RMLUI
#include "../rmlui/pipeline.hpp"
#endif
#include "../webgpu/backend.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <tracy/Tracy.hpp>

namespace aurora::gfx {
using namespace detail;
using webgpu::g_device;
using webgpu::g_queue;

namespace {
constexpr Module Log{"aurora::gfx"};
WGPURenderPipeline g_currentPipeline = nullptr;

void apply_viewport(const wgpu::RenderPassEncoder& pass, const Viewport& vp) {
  const float minDepth = gx::UseReversedZ ? 1.f - vp.zfar : vp.znear;
  const float maxDepth = gx::UseReversedZ ? 1.f - vp.znear : vp.zfar;
  pass.SetViewport(vp.left, vp.top, vp.width, vp.height, minDepth, maxDepth);
}

void apply_scissor(const wgpu::RenderPassEncoder& pass, const ClipRect& sc, const TextureExtent& size) {
  const auto x = std::clamp(static_cast<uint32_t>(sc.x), 0u, size.width);
  const auto y = std::clamp(static_cast<uint32_t>(sc.y), 0u, size.height);
  const auto w = std::clamp(static_cast<uint32_t>(sc.width), 0u, size.width - x);
  const auto h = std::clamp(static_cast<uint32_t>(sc.height), 0u, size.height - y);
  pass.SetScissorRect(x, y, w, h);
}

DrawContext make_draw_context(const RenderPass& passInfo) {
  auto& res = resources();
  return {
      .device = g_device,
      .queue = g_queue,
      .vertexBuffer = res.vertexBuffer,
      .indexBuffer = res.indexBuffer,
      .uniformBuffer = res.uniformBuffer,
      .storageBuffer = res.storageBuffer,
      .layout = passInfo.target_layout(),
  };
}

void render_custom_draw(const CustomDrawCommand& draw, const wgpu::RenderPassEncoder& pass,
                        const RenderPass& passInfo) {
  const auto drawType = find_runtime_draw_type(draw.type);
  if (!drawType) {
    // Unregistered between record and replay; the command is a no-op.
    return;
  }

  const auto context = make_draw_context(passInfo);
  drawType->draw(context, pass, draw.payload.data(), draw.payloadSize, drawType->userdata);
}

void execute_encoder_task(wgpu::CommandEncoder& cmd, webgpu::FrameState& frame, const EncoderTask& task) {
  const auto taskType = find_runtime_encoder_task_type(task.type);
  if (!taskType) {
    // Unregistered between record and encode; the task is a no-op.
    return;
  }

  auto& res = resources();
  const EncoderTaskContext context{
      .device = g_device,
      .queue = g_queue,
      .vertexBuffer = res.vertexBuffer,
      .indexBuffer = res.indexBuffer,
      .uniformBuffer = res.uniformBuffer,
      .storageBuffer = res.storageBuffer,
  };
  taskType->callback(context, cmd, task.payload.data(), task.payloadSize, taskType->userdata);
  if (taskType->afterSubmit != nullptr) {
    const auto payload = task.payload;
    const auto payloadSize = task.payloadSize;
    const auto callback = taskType->afterSubmit;
    const auto userdata = taskType->userdata;
    frame.afterSubmitCallbacks.emplace_back([payload, payloadSize, callback, userdata] {
      const EncoderTaskCompletionContext completionContext{
          .device = g_device,
          .queue = g_queue,
      };
      callback(completionContext, payload.data(), payloadSize, userdata);
    });
  }
}

void render_pass(const wgpu::RenderPassEncoder& pass, FramePacket& frame, RenderPass& passInfo) {
  ZoneScoped;
  g_currentPipeline = nullptr;
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> lastDebugGroupStack;
#endif
  Viewport currentViewport{};
  ClipRect currentScissor{};
  bool hasViewport = false;
  bool hasScissor = false;

  // Bind bind group for the whole pass
  pass.SetBindGroup(0, resources().staticBindGroup);
  pass.SetBindGroup(2, gx::g_emptyTextureBindGroup);

  for (auto& cmd : passInfo.commands) {
#ifdef AURORA_GFX_DEBUG_GROUPS
    {
      size_t firstDiff = lastDebugGroupStack.size();
      for (size_t i = 0; i < lastDebugGroupStack.size(); ++i) {
        if (i >= cmd.debugGroupStack.size() || cmd.debugGroupStack[i] != lastDebugGroupStack[i]) {
          firstDiff = i;
          break;
        }
      }
      for (size_t i = firstDiff; i < lastDebugGroupStack.size(); ++i) {
        pass.PopDebugGroup();
      }
      for (size_t i = firstDiff; i < cmd.debugGroupStack.size(); ++i) {
        pass.PushDebugGroup(cmd.debugGroupStack[i].c_str());
      }
      lastDebugGroupStack = cmd.debugGroupStack;
    }
#endif
    switch (cmd.type) {
    case CommandType::SetViewport: {
      const auto& vp = cmd.data.setViewport;
      apply_viewport(pass, vp);
      currentViewport = vp;
      hasViewport = true;
    } break;
    case CommandType::SetScissor: {
      const auto& sc = cmd.data.setScissor;
      apply_scissor(pass, sc, passInfo.colorAttachments[SceneColorAttachmentIndex].size);
      currentScissor = sc;
      hasScissor = true;
    } break;
    case CommandType::Draw: {
      auto& draw = cmd.data.draw;
      switch (draw.kind) {
      case DrawKind::Gx:
        gx::render(inline_payload<gx::DrawData>(draw.payload.data()), pass);
        break;
      case DrawKind::Clear:
        clear::render(inline_payload<clear::DrawData>(draw.payload.data()), pass,
                      passInfo.colorAttachments[SceneColorAttachmentIndex].size);
        break;
      case DrawKind::RmlUi:
#ifdef AURORA_ENABLE_RMLUI
        rmlui::render(inline_payload<rmlui::DrawData>(draw.payload.data()), pass);
#endif
        break;
      }
    } break;
    case CommandType::CustomDraw: {
      render_custom_draw(cmd.data.customDraw, pass, passInfo);
      g_currentPipeline = nullptr;
      pass.SetBindGroup(0, resources().staticBindGroup);
      pass.SetBindGroup(2, gx::g_emptyTextureBindGroup);
      if (hasViewport) {
        apply_viewport(pass, currentViewport);
      }
      if (hasScissor) {
        apply_scissor(pass, currentScissor, passInfo.colorAttachments[SceneColorAttachmentIndex].size);
      }
    } break;
    case CommandType::DebugMarker: {
#if defined(AURORA_GFX_DEBUG_GROUPS)
      pass.InsertDebugMarker(wgpu::StringView(frame.debugMarkers[cmd.data.debugMarkerIndex]));
#endif
    } break;
    }
  }

#ifdef AURORA_GFX_DEBUG_GROUPS
  for (size_t i = 0; i < lastDebugGroupStack.size(); ++i) {
    pass.PopDebugGroup();
  }
#endif
}

void render(wgpu::CommandEncoder& cmd, FramePacket& frame, RenderPass& passInfo, uint32_t passIndex) {
  ZoneScoped;
  if (!passInfo.sealed) {
    return;
  }

  for (const auto& conv : passInfo.paletteConvs) {
    tex_palette_conv::run(cmd, conv);
  }
  if (passInfo.discardable) {
    // This pass has no effect and can be safely discarded (e.g. an empty EFB segment between two back-to-back pass
    // breaks, or an unresolved offscreen pass).
    return;
  }

  // A target left null is the EFB's; the EFB resolves to its single-sample copy when multisampled, and copies read
  // that. A pass begin_color_pass opened draws to RmlUi's views instead, and has nothing to copy from.
  const auto* external = passInfo.backendTargets.get();
  const auto& sceneColor = passInfo.colorAttachments[SceneColorAttachmentIndex];
  const bool efb = external == nullptr && !sceneColor.target;
  const bool resolveEfb = efb && passInfo.msaaSamples > 1;
  wgpu::Texture copySourceTexture;
  wgpu::TextureView copySourceView;
  if (efb) {
    const auto& source = resolveEfb ? webgpu::g_frameBufferResolved : webgpu::g_frameBuffer;
    copySourceTexture = source.texture;
    copySourceView = source.view;
  } else if (external == nullptr) {
    copySourceTexture = sceneColor.target->gpu->texture;
    copySourceView = sceneColor.target->gpu->attachmentTextureView;
  }
  wgpu::TextureView depthStencilView;
  wgpu::TextureView copySourceDepthView;
  if (external != nullptr) {
    depthStencilView = external->depthStencilView;
  } else {
    depthStencilView =
        passInfo.depthTarget ? passInfo.depthTarget->gpu->attachmentTextureView : webgpu::g_depthBuffer.view;
    copySourceDepthView = depthStencilView;
  }

  std::array<wgpu::RenderPassColorAttachment, MaxColorAttachments> attachments{};
  for (uint32_t i = 0; i < passInfo.colorAttachmentCount; ++i) {
    const auto& source = passInfo.colorAttachments[i];
    wgpu::TextureView view;
    wgpu::TextureView resolveView;
    if (external != nullptr) {
      view = external->colorView;
      resolveView = external->resolveView;
    } else if (source.target) {
      view = source.target->gpu->attachmentTextureView;
    } else if (source.semantic == ColorAttachmentSemantic::Normal) {
      view = webgpu::g_normalBuffer.view;
    } else {
      view = webgpu::g_frameBuffer.view;
      resolveView = resolveEfb ? webgpu::g_frameBufferResolved.view : nullptr;
    }
    const auto loadOp = external != nullptr ? external->colorLoadOp : wgpu::LoadOp::Undefined;
    attachments[i] = {
        .view = view,
        .resolveTarget = resolveView,
        .loadOp =
            loadOp != wgpu::LoadOp::Undefined ? loadOp : (source.clear ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load),
        .storeOp = external != nullptr ? external->colorStoreOp : wgpu::StoreOp::Store,
        .clearValue =
            {
                .r = source.clearValue.x(),
                .g = source.clearValue.y(),
                .b = source.clearValue.z(),
                .a = source.clearValue.w(),
            },
    };
  }
  wgpu::RenderPassDepthStencilAttachment depthStencilAttachment{};
  const wgpu::RenderPassDepthStencilAttachment* depthStencilAttachmentPtr = nullptr;
  if (depthStencilView) {
    // Only begin_color_pass gives load and store operations, and only its passes have a stencil.
    const auto depthLoadOp = external != nullptr ? external->depthLoadOp : wgpu::LoadOp::Undefined;
    const bool hasStencil = passInfo.hasStencil && external != nullptr;
    depthStencilAttachment = {
        .view = depthStencilView,
        .depthLoadOp = passInfo.hasDepth ? (depthLoadOp != wgpu::LoadOp::Undefined
                                                ? depthLoadOp
                                                : (passInfo.clearDepth ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load))
                                         : wgpu::LoadOp::Undefined,
        .depthStoreOp = passInfo.hasDepth ? (external != nullptr ? external->depthStoreOp : wgpu::StoreOp::Store)
                                          : wgpu::StoreOp::Undefined,
        .depthClearValue = passInfo.clearDepthValue,
        .stencilLoadOp = hasStencil ? external->stencilLoadOp : wgpu::LoadOp::Undefined,
        .stencilStoreOp = hasStencil ? external->stencilStoreOp : wgpu::StoreOp::Undefined,
        .stencilClearValue = external != nullptr ? external->stencilClearValue : 0,
    };
    depthStencilAttachmentPtr = &depthStencilAttachment;
  }
  const auto label = passInfo.label.empty() ? fmt::format("Render pass {}", passIndex)
                                            : fmt::format("{} {}", passInfo.label, passIndex);
  const wgpu::RenderPassDescriptor renderPassDescriptor{
      .label = label.c_str(),
      .colorAttachmentCount = passInfo.colorAttachmentCount,
      .colorAttachments = attachments.data(),
      .depthStencilAttachment = depthStencilAttachmentPtr,
      .timestampWrites = webgpu::gpu_prof::pass_writes(label),
  };

  auto pass = cmd.BeginRenderPass(&renderPassDescriptor);
  render_pass(pass, frame, passInfo);
  pass.End();

  if (passInfo.captureDepthSnapshot) {
    depth_peek::encode_frame_snapshot(cmd, copySourceDepthView, {sceneColor.size.width, sceneColor.size.height, 1},
                                      passInfo.msaaSamples);
  }

  if (passInfo.resolveTarget) {
    const auto& dstSize = passInfo.resolveTarget->size;
    const bool needsConversion = tex_copy_conv::needs_conversion(passInfo.resolveFormat);
    const bool needsScaling = dstSize.width != static_cast<uint32_t>(passInfo.resolveRect.width) ||
                              dstSize.height != static_cast<uint32_t>(passInfo.resolveRect.height);
    const bool isDepth = gx::is_depth_format(passInfo.resolveFormat);
    if (isDepth && passInfo.msaaSamples > 1) {
      Log.fatal("Depth tex copies from multisampled EFB targets are not supported");
    }
    const tex_copy_conv::ConvRequest convReq{
        .fmt = passInfo.resolveFormat,
        .srcFmt = passInfo.resolveSourceFormat,
        .srcView = isDepth ? copySourceDepthView : copySourceView,
        .uniformRange = passInfo.resolveUniformRange,
        .dst = passInfo.resolveTarget,
        .sampleFilter = needsScaling ? tex_copy_conv::SampleFilter::Linear : tex_copy_conv::SampleFilter::Nearest,
    };
    const bool needsOpaqueAlpha = !isDepth && !gx::efb_has_alpha(convReq.srcFmt);
    const bool sameFormat =
        passInfo.resolveTarget->format == passInfo.colorAttachments[SceneColorAttachmentIndex].format;
    if (needsConversion) {
      tex_copy_conv::run(cmd, convReq);
    } else if (needsScaling || needsOpaqueAlpha || !sameFormat) {
      tex_copy_conv::blit(cmd, convReq);
    } else {
      const webgpu::gpu_prof::Zone zone{cmd, "EFB copy"};
      const wgpu::TexelCopyTextureInfo src{
          .texture = copySourceTexture,
          .origin =
              wgpu::Origin3D{
                  .x = static_cast<uint32_t>(passInfo.resolveRect.x),
                  .y = static_cast<uint32_t>(passInfo.resolveRect.y),
              },
      };
      const wgpu::TexelCopyTextureInfo dst{
          .texture = passInfo.resolveTarget->gpu->texture,
      };
      const wgpu::Extent3D size{
          .width = static_cast<uint32_t>(passInfo.resolveRect.width),
          .height = static_cast<uint32_t>(passInfo.resolveRect.height),
          .depthOrArrayLayers = 1,
      };
      cmd.CopyTextureToTexture(&src, &dst, &size);
    }
  }

  if (passInfo.snapshotColor) {
    const webgpu::gpu_prof::Zone zone{cmd, "Pass snapshot"};
    const wgpu::TexelCopyTextureInfo src{
        .texture = copySourceTexture,
    };
    const wgpu::TexelCopyTextureInfo dst{
        .texture = passInfo.snapshotColor->gpu->texture,
    };
    const wgpu::Extent3D size{
        .width = passInfo.colorAttachments[SceneColorAttachmentIndex].size.width,
        .height = passInfo.colorAttachments[SceneColorAttachmentIndex].size.height,
        .depthOrArrayLayers = 1,
    };
    cmd.CopyTextureToTexture(&src, &dst, &size);
  }
  if (passInfo.snapshotDepth) {
    tex_copy_conv::snapshot_depth(cmd, copySourceDepthView, passInfo.msaaSamples,
                                  passInfo.snapshotDepth->gpu->attachmentTextureView);
  }
  if (passInfo.snapshotNormal) {
    // Only the EFB has a normal attachment.
    const webgpu::gpu_prof::Zone zone{cmd, "Normal snapshot"};
    const wgpu::TexelCopyTextureInfo src{
        .texture = webgpu::g_normalBuffer.texture,
    };
    const wgpu::TexelCopyTextureInfo dst{
        .texture = passInfo.snapshotNormal->gpu->texture,
    };
    const wgpu::Extent3D size{
        .width = passInfo.colorAttachments[SceneColorAttachmentIndex].size.width,
        .height = passInfo.colorAttachments[SceneColorAttachmentIndex].size.height,
        .depthOrArrayLayers = 1,
    };
    cmd.CopyTextureToTexture(&src, &dst, &size);
  }
}

constexpr uint64_t VertexStagingOffset = 0;
constexpr uint64_t UniformStagingOffset = VertexStagingOffset + VertexBufferSize;
constexpr uint64_t IndexStagingOffset = UniformStagingOffset + UniformBufferSize;
constexpr uint64_t StorageStagingOffset = IndexStagingOffset + IndexBufferSize;
constexpr uint64_t TextureUploadStagingOffset = StorageStagingOffset + StorageBufferSize;

// Staging copies start and end on 64 KiB boundaries, clamped to the pool. Mesa's hasvk runs vkCmdCopyBuffer as a
// blorp blit, and on that hardware copies off those boundaries corrupted vertex data and cost CPU in its extent search.
constexpr uint32_t StagingCopyAlign = 64 * 1024;

constexpr uint32_t align_down_copy_offset(uint32_t value) noexcept { return value & ~(StagingCopyAlign - 1); }

void copy_staging_buffer_range(wgpu::CommandEncoder& cmd, const webgpu::FrameState& frame, uint32_t& copied,
                               uint32_t highWater, uint64_t stagingOffset, uint64_t poolSize, const wgpu::Buffer& dst) {
  if (highWater <= copied) {
    return;
  }
  const uint32_t copyStart = align_down_copy_offset(copied);
  const uint32_t copyEnd =
      static_cast<uint32_t>(std::min<uint64_t>(AURORA_ALIGN(uint64_t{highWater}, StagingCopyAlign), poolSize));
  cmd.CopyBufferToBuffer(staging_buffer(frame.stagingBuffer), stagingOffset + copyStart, dst, copyStart,
                         copyEnd - copyStart);
  copied = highWater;
}

bool needs_staging_copy(const webgpu::FrameState& frame, const FrameOp& op) {
  const auto& highWater = op.highWater;
  if (highWater.verts > frame.copied.verts || highWater.uniforms > frame.copied.uniforms ||
      highWater.indices > frame.copied.indices || highWater.storage > frame.copied.storage) {
    return true;
  }
  if constexpr (UseTextureBuffer) {
    return op.textureUploads.size() > frame.copied.textureUploadCount;
  }
  return false;
}

void copy_staging_to_high_water(wgpu::CommandEncoder& cmd, webgpu::FrameState& frame, const FrameOp& op) {
  if (!needs_staging_copy(frame, op)) {
    return;
  }
  const webgpu::gpu_prof::Zone zone{cmd, "Staging copies"};
  const auto& highWater = op.highWater;
  auto& res = resources();
  copy_staging_buffer_range(cmd, frame, frame.copied.verts, highWater.verts, VertexStagingOffset, VertexBufferSize,
                            res.vertexBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.uniforms, highWater.uniforms, UniformStagingOffset,
                            UniformBufferSize, res.uniformBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.indices, highWater.indices, IndexStagingOffset, IndexBufferSize,
                            res.indexBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.storage, highWater.storage, StorageStagingOffset,
                            StorageBufferSize, res.storageBuffer);

  if constexpr (UseTextureBuffer) {
    for (size_t i = frame.copied.textureUploadCount; i < op.textureUploads.size(); ++i) {
      const auto& item = *op.textureUploads[i];
      const wgpu::TexelCopyBufferInfo buf{
          .layout =
              wgpu::TexelCopyBufferLayout{
                  .offset = item.overflow ? 0 : item.offset + TextureUploadStagingOffset,
                  .bytesPerRow = AURORA_ALIGN(item.bytesPerRow, backend::limits().textureRowAlignment),
                  .rowsPerImage = item.rowsPerImage,
              },
          .buffer = item.overflow ? item.overflow->buffer : staging_buffer(frame.stagingBuffer),
      };
      const wgpu::TexelCopyTextureInfo tex{
          .texture = item.texture->texture,
          .mipLevel = item.mipLevel,
      };
      const wgpu::Extent3D size{item.width, item.height, item.layers};
      cmd.CopyBufferToTexture(&buf, &tex, &size);
    }
    frame.copied.textureUpload = highWater.textureUpload;
    frame.copied.textureUploadCount = op.textureUploads.size();
  }
}
} // namespace

namespace backend {
void encode(FramePacket& frame, const FrameOp& op) noexcept {
  auto& state = webgpu::frame_state(frame.slot);
  auto& cmd = state.encoder;
  copy_staging_to_high_water(cmd, state, op);
  switch (op.type) {
  case FrameOpType::RenderPass:
    if (op.renderPass != nullptr) {
      render(cmd, frame, *op.renderPass, op.index);
    }
    break;
  case FrameOpType::TextureCopy:
    if (op.textureCopy != nullptr) {
      const webgpu::gpu_prof::Zone zone{cmd, "Texture copy"};
      cmd.CopyTextureToTexture(&op.textureCopy->src, &op.textureCopy->dst, &op.textureCopy->size);
    }
    break;
  case FrameOpType::EncoderTask:
    if (op.encoderTask != nullptr) {
      execute_encoder_task(cmd, state, *op.encoderTask);
    }
    break;
  }
}
} // namespace backend

void bind_pipeline(const wgpu::RenderPipeline& pipeline, const wgpu::RenderPassEncoder& pass) {
  if (pipeline.Get() == g_currentPipeline) {
    return;
  }
  pass.SetPipeline(pipeline);
  g_currentPipeline = pipeline.Get();
}

bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass) {
  CompiledPipeline pipeline;
  if (!get_pipeline(ref, pipeline)) {
    return false;
  }
  AURORA_ASSERT(!pipeline.prepass, "Prepass pipelines require individual binding");
  bind_pipeline(pipeline.main->pipeline, pass);
  return true;
}
} // namespace aurora::gfx
