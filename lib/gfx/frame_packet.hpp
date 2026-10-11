#pragma once

#include "pipeline_cache.hpp"
#include "types.hpp"
#include "tex_palette_conv.hpp"
#include "texture.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

namespace aurora::gfx::detail {

struct StagingHighWater {
  uint32_t verts = 0;
  uint32_t uniforms = 0;
  uint32_t indices = 0;
  uint32_t storage = 0;
  uint32_t textureUpload = 0;
  size_t textureUploadCount = 0;
};

struct CustomDrawCommand {
  DrawTypeId type = 0;
  uint32_t payloadSize = 0;
  alignas(std::max_align_t) std::array<std::byte, InlineDrawPayloadSize> payload{};
};

// Which renderer a draw's payload is for; the backend draws each kind.
enum class DrawKind : uint8_t {
  Gx,    // gx::DrawData
  Clear, // clear::DrawData
  RmlUi, // rmlui::DrawData, recorded only with RmlUi on
};

struct DrawCommand {
  DrawKind kind = DrawKind::Gx;
  alignas(std::max_align_t) std::array<std::byte, InlineDrawPayloadSize> payload{};
};

template <class T>
concept InlinePayload =
    std::is_trivially_copyable_v<T> && std::is_aggregate_v<T> && std::is_trivially_destructible_v<T>;

template <InlinePayload T>
T& inline_payload(void* ptr) noexcept {
  // C++20 equivalent of C++23's std::start_lifetime_as
  return *std::launder(static_cast<T*>(std::memmove(ptr, ptr, sizeof(T))));
}

enum class CommandType {
  SetViewport,
  SetScissor,
  Draw,
  CustomDraw,
  DebugMarker,
};

struct Command {
  CommandType type;
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> debugGroupStack;
#endif
  union Data {
    Viewport setViewport;
    ClipRect setScissor;
    DrawCommand draw;
    CustomDrawCommand customDraw;
    size_t debugMarkerIndex;
  } data;
};

using CommandList = std::vector<Command>;

struct RenderPass {
  struct ColorAttachment {
    ColorAttachmentSemantic semantic = ColorAttachmentSemantic::Auxiliary;
    TextureFormat format = TextureFormat::Undefined;
    TextureExtent size{};
    // Null for the EFB's attachment of this semantic, which the backend owns.
    TextureHandle target;
    Vec4<float> clearValue{0.f, 0.f, 0.f, 0.f};
    bool clear = true;
  };

  std::string label;
  std::array<ColorAttachment, MaxColorAttachments> colorAttachments;
  uint32_t colorAttachmentCount = 0;
  // Null for the EFB's depth.
  TextureHandle depthTarget;
  TextureFormat depthStencilFormat = TextureFormat::Undefined;
  // Set when a backend's own API opened the pass on its own targets (RmlUi's on Dawn), which replace the ones above.
  std::shared_ptr<const backend::PassTargets> backendTargets;
  uint32_t msaaSamples = 1;

  TextureHandle resolveTarget;
  GXTexFmt resolveFormat = GX_TF_RGBA8;
  GXPixelFmt resolveSourceFormat = GX_PF_RGBA6_Z24;
  ClipRect resolveRect;
  Range resolveUniformRange;
  // resolve_pass's copies of the pass's colour, depth (R32Float) and normal targets.
  TextureHandle snapshotColor;
  TextureHandle snapshotDepth;
  TextureHandle snapshotNormal;
  float clearDepthValue = 1.f;
  CommandList commands;
  bool clearDepth = true;
  bool hasDepth = true;
  bool hasStencil = false;
  bool hasDraws = false;
  bool discardable = false;
  bool captureDepthSnapshot = false;
  bool sealed = false;
  std::vector<tex_palette_conv::ConvRequest> paletteConvs;

  RenderTargetLayout target_layout() const noexcept;
  bool has_consumer() const { return resolveTarget || snapshotColor || snapshotDepth || snapshotNormal; }
  bool has_content() const {
    if (hasDraws || clearDepth) {
      return true;
    }
    for (uint32_t i = 0; i < colorAttachmentCount; ++i) {
      if (colorAttachments[i].clear) {
        return true;
      }
    }
    return false;
  }
};

inline void finalize_render_target_layout(RenderTargetLayout& layout) noexcept {
  Hasher hasher{};
  hasher.update(layout.colorAttachmentCount);
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    hasher.update(layout.colorAttachments[i].semantic);
    hasher.update(layout.colorAttachments[i].format);
  }
  hasher.update(static_cast<uint32_t>(layout.depthStencilFormat));
  hasher.update(layout.sampleCount);
  layout.key = hasher.digest();
}

inline RenderTargetLayout RenderPass::target_layout() const noexcept {
  RenderTargetLayout layout{
      .colorAttachmentCount = colorAttachmentCount,
      .depthStencilFormat = hasDepth || hasStencil ? depthStencilFormat : TextureFormat::Undefined,
      .sampleCount = msaaSamples,
  };
  for (uint32_t i = 0; i < colorAttachmentCount; ++i) {
    layout.colorAttachments[i] = {
        .semantic = colorAttachments[i].semantic,
        .format = colorAttachments[i].format,
        .width = colorAttachments[i].size.width,
        .height = colorAttachments[i].size.height,
    };
  }
  finalize_render_target_layout(layout);
  return layout;
}

struct EncoderTask {
  EncoderTaskId type = InvalidEncoderTask;
  std::array<uint8_t, InlineDrawPayloadSize> payload{};
  uint32_t payloadSize = 0;
};

enum class FrameOpType : uint8_t {
  RenderPass,
  TextureCopy,
  EncoderTask,
};

struct FrameOp {
  FrameOpType type = FrameOpType::RenderPass;
  uint32_t index = 0;
  RenderPass* renderPass = nullptr;
  const backend::TextureCopy* textureCopy = nullptr;
  EncoderTask* encoderTask = nullptr;
  StagingHighWater highWater;
  std::vector<const TextureUpload*> textureUploads;
};

using RenderPassList = std::deque<RenderPass>;

struct FramePacket {
  RenderPassList renderPasses;
  std::deque<std::shared_ptr<const backend::TextureCopy>> textureCopies;
  std::deque<EncoderTask> encoderTasks;
  std::deque<FrameOp> ops;
  std::deque<TextureUpload> textureUploads;
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> debugMarkers;
#endif
  ByteBuffer verts;
  ByteBuffer uniforms;
  ByteBuffer indices;
  ByteBuffer storage;
  ByteBuffer textureUpload;
  uint64_t frameId = 0;
  uint32_t frameIndex = 0;
  // The frame slot, under which a backend keeps its per-frame state.
  size_t slot = 0;
  // Where the backend writes the presented frame as an image (aurora_capture_frame); empty for none.
  std::string capturePath;
  AuroraStats stats{};
};

} // namespace aurora::gfx::detail
