#include "shader_info.hpp"

#include "../gfx/recording.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>

#include <tracy/Tracy.hpp>

namespace aurora::gx {
namespace {
constexpr Module Log{"aurora::gx"};

bool is_alpha_bump_channel(GXChannelID id) { return id == GX_ALPHA_BUMP || id == GX_ALPHA_BUMPN; }

Vec4<float> texture_size_bias(const gfx::TextureBind& tex) {
  auto width = static_cast<float>(tex.texObj.width());
  auto height = static_cast<float>(tex.texObj.height());
  float vpBias = 0.f;
  if (tex.ref && tex.ref->hasArbitraryMips) {
    const float viewportScale =
        std::min(g_gxState.renderViewport.width / std::max(g_gxState.logicalViewport.width, 1.f),
                 g_gxState.renderViewport.height / std::max(g_gxState.logicalViewport.height, 1.f));
    const float replacementScale = static_cast<float>(tex.ref->size.width) / std::max(width, 1.f);
    vpBias = std::log2(viewportScale / std::max(replacementScale, 0.001f));
  }
  return {width, height, tex.texObj.lod_bias() + vpBias, 0.0f};
}

void color_arg_reg_info(GXTevColorArg arg, const TevStage& stage, ShaderInfo& info) {
  switch (arg) {
  case GX_CC_CPREV:
  case GX_CC_APREV:
    if (!info.writesTevReg.test(GX_TEVPREV)) {
      info.loadsTevReg.set(GX_TEVPREV);
    }
    break;
  case GX_CC_C0:
  case GX_CC_A0:
    if (!info.writesTevReg.test(GX_TEVREG0)) {
      info.loadsTevReg.set(GX_TEVREG0);
    }
    break;
  case GX_CC_C1:
  case GX_CC_A1:
    if (!info.writesTevReg.test(GX_TEVREG1)) {
      info.loadsTevReg.set(GX_TEVREG1);
    }
    break;
  case GX_CC_C2:
  case GX_CC_A2:
    if (!info.writesTevReg.test(GX_TEVREG2)) {
      info.loadsTevReg.set(GX_TEVREG2);
    }
    break;
  case GX_CC_TEXC:
  case GX_CC_TEXA:
    if (stage.texMapId != GX_TEXMAP_NULL) {
      CHECK(stage.texCoordId != GX_TEXCOORD_NULL, "tex coord not bound");
      info.sampledTexCoords.set(stage.texCoordId);
      info.sampledTextures.set(stage.texMapId);
    }
    break;
  case GX_CC_RASC:
  case GX_CC_RASA:
    if (stage.channelId != GX_COLOR_NULL && stage.channelId != GX_COLOR_ZERO &&
        !is_alpha_bump_channel(stage.channelId)) {
      info.sampledColorChannels.set(color_channel(stage.channelId));
    }
    break;
  case GX_CC_KONST:
    switch (stage.kcSel) {
    case GX_TEV_KCSEL_K0:
    case GX_TEV_KCSEL_K0_R:
    case GX_TEV_KCSEL_K0_G:
    case GX_TEV_KCSEL_K0_B:
    case GX_TEV_KCSEL_K0_A:
      info.sampledKColors.set(0);
      break;
    case GX_TEV_KCSEL_K1:
    case GX_TEV_KCSEL_K1_R:
    case GX_TEV_KCSEL_K1_G:
    case GX_TEV_KCSEL_K1_B:
    case GX_TEV_KCSEL_K1_A:
      info.sampledKColors.set(1);
      break;
    case GX_TEV_KCSEL_K2:
    case GX_TEV_KCSEL_K2_R:
    case GX_TEV_KCSEL_K2_G:
    case GX_TEV_KCSEL_K2_B:
    case GX_TEV_KCSEL_K2_A:
      info.sampledKColors.set(2);
      break;
    case GX_TEV_KCSEL_K3:
    case GX_TEV_KCSEL_K3_R:
    case GX_TEV_KCSEL_K3_G:
    case GX_TEV_KCSEL_K3_B:
    case GX_TEV_KCSEL_K3_A:
      info.sampledKColors.set(3);
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }
}

void alpha_arg_reg_info(GXTevAlphaArg arg, const TevStage& stage, ShaderInfo& info) {
  switch (arg) {
  case GX_CA_APREV:
    if (!info.writesTevReg.test(GX_TEVPREV)) {
      info.loadsTevReg.set(GX_TEVPREV);
    }
    break;
  case GX_CA_A0:
    if (!info.writesTevReg.test(GX_TEVREG0)) {
      info.loadsTevReg.set(GX_TEVREG0);
    }
    break;
  case GX_CA_A1:
    if (!info.writesTevReg.test(GX_TEVREG1)) {
      info.loadsTevReg.set(GX_TEVREG1);
    }
    break;
  case GX_CA_A2:
    if (!info.writesTevReg.test(GX_TEVREG2)) {
      info.loadsTevReg.set(GX_TEVREG2);
    }
    break;
  case GX_CA_TEXA:
    if (stage.texMapId != GX_TEXMAP_NULL) {
      CHECK(stage.texCoordId != GX_TEXCOORD_NULL, "tex coord not bound");
      info.sampledTexCoords.set(stage.texCoordId);
      info.sampledTextures.set(stage.texMapId);
    }
    break;
  case GX_CA_RASA:
    if (stage.channelId != GX_COLOR_NULL && stage.channelId != GX_COLOR_ZERO &&
        !is_alpha_bump_channel(stage.channelId)) {
      info.sampledColorChannels.set(color_channel(stage.channelId));
    }
    break;
  case GX_CA_KONST:
    switch (stage.kaSel) {
    case GX_TEV_KASEL_K0_R:
    case GX_TEV_KASEL_K0_G:
    case GX_TEV_KASEL_K0_B:
    case GX_TEV_KASEL_K0_A:
      info.sampledKColors.set(0);
      break;
    case GX_TEV_KASEL_K1_R:
    case GX_TEV_KASEL_K1_G:
    case GX_TEV_KASEL_K1_B:
    case GX_TEV_KASEL_K1_A:
      info.sampledKColors.set(1);
      break;
    case GX_TEV_KASEL_K2_R:
    case GX_TEV_KASEL_K2_G:
    case GX_TEV_KASEL_K2_B:
    case GX_TEV_KASEL_K2_A:
      info.sampledKColors.set(2);
      break;
    case GX_TEV_KASEL_K3_R:
    case GX_TEV_KASEL_K3_G:
    case GX_TEV_KASEL_K3_B:
    case GX_TEV_KASEL_K3_A:
      info.sampledKColors.set(3);
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }
}

f32 tex_offset(GXTexOffset offs) noexcept {
  switch (offs) {
  default:
  case GX_TO_ZERO:
    return 0.f;
  case GX_TO_SIXTEENTH:
    return 1.f / 16.f;
  case GX_TO_EIGHTH:
    return 1.f / 8.f;
  case GX_TO_FOURTH:
    return 1.f / 4.f;
  case GX_TO_HALF:
    return 1.f / 2.f;
  case GX_TO_ONE:
    return 1.f;
  }
}

u32 point_texcoord_mask() noexcept {
  u32 mask = 0;
  for (int i = 0; i < MaxTexCoord; ++i) {
    if (g_gxState.texCoordScales[i].pointOffset) {
      mask |= 1 << i;
    }
  }
  return mask;
}

u32 line_texcoord_mask() noexcept {
  u32 mask = 0;
  for (int i = 0; i < MaxTexCoord; ++i) {
    if (g_gxState.texCoordScales[i].lineOffset) {
      mask |= 1 << i;
    }
  }
  return mask;
}

// The Z texture unit consumes the texture sample of the last TEV stage.
// Returns that stage index, or -1 when the fragment shader must not write
// frag_depth (ztex disabled, or a case we don't implement).
s8 ztex_source_stage(const ShaderConfig& config) noexcept {
  if (config.zTexOp == GX_ZT_DISABLE || config.tevStageCount == 0) {
    return -1;
  }
  if (config.zCompLocBeforeTex) {
    /* Compare-before-texture: the depth buffer gets the polygon's own depth
     * and the Z texture only feeds fog. HSD_EraseRect selects this, and
     * emitting frag_depth there replaced its erase quad's near-far depth
     * with 0xFF0000/0xFFFFFF = 0.996, which against a 0.1 near plane cut
     * every Adventure cutscene off at ~25 world units. */
    return -1;
  }
  if (config.zTexFmt == 1 /* U16 */) {
    // Unused by any known title on this port, and the GX_TF_Z16 copy
    // conversion in tex_copy_conv.cpp packs its two bytes the other way round
    // from Dolphin's U16 ztex coefficients, so the byte order here is
    // unverified. Refuse rather than guess.
    static bool warned = false;
    if (!warned) {
      warned = true;
      Log.warn("Z texture format U16 is not implemented; ignoring ztex");
    }
    return -1;
  }
  const auto& stage = config.tevStages[config.tevStageCount - 1];
  if (stage.texMapId == GX_TEXMAP_NULL || stage.texCoordId == GX_TEXCOORD_NULL) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      Log.warn("Z texture enabled but the last TEV stage has no texture; ignoring ztex");
    }
    return -1;
  }
  return static_cast<s8>(config.tevStageCount - 1);
}
} // namespace

ShaderInfo build_shader_info(const ShaderConfig& config) noexcept {
  ZoneScoped;

  ShaderInfo info{};

  if (config.lineMode != 0) {
    info.lineMode = config.lineMode;
  }

  for (int attr = 0; attr < config.attrs.size(); attr++) {
    const auto attrType = config.attrs[attr].attrType;
    if ((attr == GX_VA_PNMTXIDX || (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX)) && attrType == GX_DIRECT) {
      info.indexAttr.set(attr);
    } else if (attrType == GX_INDEX8 || attrType == GX_INDEX16) {
      info.indexAttr.set(attr);
    }
  }

  for (int i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    // Color pass
    color_arg_reg_info(stage.colorPass.a, stage, info);
    color_arg_reg_info(stage.colorPass.b, stage, info);
    color_arg_reg_info(stage.colorPass.c, stage, info);
    color_arg_reg_info(stage.colorPass.d, stage, info);
    info.writesTevReg.set(stage.colorOp.outReg);

    // Alpha pass
    alpha_arg_reg_info(stage.alphaPass.a, stage, info);
    alpha_arg_reg_info(stage.alphaPass.b, stage, info);
    alpha_arg_reg_info(stage.alphaPass.c, stage, info);
    alpha_arg_reg_info(stage.alphaPass.d, stage, info);
    if (!info.writesTevReg.test(stage.alphaOp.outReg)) {
      // If we're writing alpha to a register that's not been
      // written to in the shader, load from uniform buffer
      info.loadsTevReg.set(stage.alphaOp.outReg);
      info.writesTevReg.set(stage.alphaOp.outReg);
    }
  }
  info.zTexStage = ztex_source_stage(config);
  if (info.zTexStage >= 0) {
    // The ztex source sample is needed even when the TEV stage discards it in
    // the colour path (the Classic intro's stage 1 does exactly that).
    const auto& stage = config.tevStages[info.zTexStage];
    info.sampledTexCoords.set(stage.texCoordId);
    info.sampledTextures.set(stage.texMapId);
  }
  for (int i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    // Skip if not enabled
    if (stage.indTexStage >= config.numIndStages) {
      continue;
    }
    const bool usesIndStage = stage.indTexMtxId != GX_ITM_OFF || is_alpha_bump_channel(stage.channelId);
    const bool usesTevTexCoord = stage.indTexMtxId != GX_ITM_OFF || stage.indTexWrapS != GX_ITW_OFF ||
                                 stage.indTexWrapT != GX_ITW_OFF || stage.indTexAddPrev;
    if (usesTevTexCoord && stage.texCoordId != GX_TEXCOORD_NULL) {
      info.sampledTexCoords.set(stage.texCoordId);
    }
    if (usesTevTexCoord && stage.texMapId != GX_TEXMAP_NULL &&
        (stage.indTexMtxId != GX_ITM_OFF || stage.indTexAddPrev)) {
      info.sampledTextures.set(stage.texMapId);
    }
    if (!usesIndStage) {
      continue;
    }
    info.usedIndStages.set(stage.indTexStage);
    const auto& indStage = config.indStages[stage.indTexStage];
    info.sampledTextures.set(indStage.texMapId);
    info.sampledTexCoords.set(indStage.texCoordId);
    info.sampledIndTextures.set(indStage.texMapId);
    // Track which indirect matrix is used
    if (stage.indTexMtxId >= GX_ITM_0 && stage.indTexMtxId <= GX_ITM_2) {
      info.usedIndTexMtxs.set(stage.indTexMtxId - GX_ITM_0);
    } else if (stage.indTexMtxId >= GX_ITM_S0 && stage.indTexMtxId <= GX_ITM_S2) {
      info.usedIndTexMtxs.set(stage.indTexMtxId - GX_ITM_S0);
    } else if (stage.indTexMtxId >= GX_ITM_T0 && stage.indTexMtxId <= GX_ITM_T2) {
      info.usedIndTexMtxs.set(stage.indTexMtxId - GX_ITM_T0);
    }
  }

  // Emboss bump needs its source texcoord generated and a light enabled
  for (int i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const auto& tcg = config.tcgs[i];
    if (tcg.type >= GX_TG_BUMP0 && tcg.type <= GX_TG_BUMP7) {
      info.sampledTexCoords.set(tcg.embossSrc);
      info.lightingEnabled = true;
    }
  }

  // GX_TG_SRTG reads a colour channel's rasterized output, so that channel is computed even when no TEV stage samples
  // it.
  for (int i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const auto& tcg = config.tcgs[i];
    if (tcg.type == GX_TG_SRTG && (tcg.src == GX_TG_COLOR0 || tcg.src == GX_TG_COLOR1)) {
      info.sampledColorChannels.set(tcg.src == GX_TG_COLOR0 ? 0 : 1);
    }
  }

  for (int i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (info.sampledColorChannels.test(i)) {
      const auto& cc = config.colorChannels[i];
      const auto& cca = config.colorChannels[i + GX_ALPHA0];
      if (cc.lightingEnabled || cca.lightingEnabled) {
        info.lightingEnabled = true;
      }
    }
  }
  for (int i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const auto& tcg = config.tcgs[i];
    if (tcg.postMtx != GX_PTIDENTITY) {
      u32 postMtxIdx = (tcg.postMtx - GX_PTTEXMTX0) / 3;
      info.usesPTTexMtx.set(postMtxIdx);
    }
  }
  if (config.fogType != GX_FOG_NONE) {
    info.usesFog = true;
  }
  // The uniform pieces the shaders read. Only lighting and emboss read the lights and normals.
  info.uniformPieces = 1 << UniformView | 1 << UniformPosMtx | 1 << UniformMaterial;
  if (info.lightingEnabled) {
    info.uniformPieces |= 1 << UniformNrmMtx | 1 << UniformLights;
  }
  return info;
}

// Fill_uniform, a piece at a time, written where the frame's uniform pool keeps it. Each piece holds all its state, as
// gx.hpp lays it out, and each of its bytes is written once and never read back: the pool can be uncached or
// write-combined GPU memory, where a piece built on the stack first would cost a write and a read more of each byte.
template <typename T>
void put(u8* piece, size_t offset, const T& value) noexcept {
  std::memcpy(piece + offset, &value, sizeof(T));
}

static void write_view_uniform(u8* out, u8 lineMode) noexcept {
  put(out, offsetof(ViewUniform, renderViewportSize),
      Vec2<float>{g_gxState.renderViewport.width, g_gxState.renderViewport.height});
  put(out, offsetof(ViewUniform, logicalViewportSize),
      Vec2<float>{g_gxState.logicalViewport.width, g_gxState.logicalViewport.height});
  f32 lineWidth = 0.f;
  f32 lineAspectY = 0.f;
  f32 lineTexOffset = 0.f;
  u32 lineTexCoordMask = 0;
  if (lineMode == 3) { // GX_POINTS
    lineWidth = static_cast<f32>(g_gxState.pointSize) / 6.f;
    lineAspectY = 1.0f;
    lineTexOffset = tex_offset(g_gxState.pointTexOffset);
    lineTexCoordMask = point_texcoord_mask();
  } else if (lineMode != 0) { // GX_LINES / GX_LINESTRIP
    lineWidth = static_cast<f32>(g_gxState.lineWidth) / 6.f;
    lineAspectY = g_gxState.lineHalfAspect ? 0.5f : 1.f;
    lineTexOffset = tex_offset(g_gxState.lineTexOffset);
    lineTexCoordMask = line_texcoord_mask();
  }
  put(out, offsetof(ViewUniform, lineWidth), lineWidth);
  put(out, offsetof(ViewUniform, lineAspectY), lineAspectY);
  put(out, offsetof(ViewUniform, lineTexOffset), lineTexOffset);
  put(out, offsetof(ViewUniform, lineTexCoordMask), lineTexCoordMask);
  Mat4x4<float> proj = g_gxState.proj;
  if constexpr (UseReversedZ) {
    proj.m2 = proj.m2 * Vec4{-1.f, -1.f, -1.f, -1.f};
  } else {
    proj.m2 = proj.m2 + proj.m3;
  }
  put(out, offsetof(ViewUniform, proj), proj);
  put(out, offsetof(ViewUniform, texMtxs), g_gxState.texMtxs);
  std::array<Vec4<float>, 8> channelColors;
  for (int i = 0; i < 2; ++i) {
    channelColors[i * 4] = g_gxState.colorChannelState[i].ambColor;
    channelColors[i * 4 + 1] = g_gxState.colorChannelState[i].matColor;
    channelColors[i * 4 + 2] = g_gxState.colorChannelState[i + GX_ALPHA0].ambColor;
    channelColors[i * 4 + 3] = g_gxState.colorChannelState[i + GX_ALPHA0].matColor;
  }
  put(out, offsetof(ViewUniform, channelColors), channelColors);
  put(out, offsetof(ViewUniform, postMtxs), g_gxState.ptTexMtxs);
  const auto& state = g_gxState.fog;
  const float logicalWidth = std::max(g_gxState.logicalViewport.width, 1.f);
  const float renderWidth = std::max(g_gxState.renderViewport.width, 1.f);
  Fog fog{
      .color = state.color,
      .a = state.a,
      .b = state.b,
      .c = state.c,
      .rangeCenter = ((static_cast<float>(state.rangeCenter) - g_gxState.logicalViewport.left) / logicalWidth) * 2.f -
                     1.f + (g_gxState.renderViewport.left / renderWidth) * 2.f,
  };
  for (u32 i = 0; i < state.rangeK.size(); ++i) {
    const u32 source = (i & ~1u) | (1u - (i & 1u));
    fog.rangeK[i / 4][i % 4] = static_cast<float>(state.rangeK[source]) / 64.f;
  }
  fog.rangeK[2][2] = fog.rangeK[2][1];
  fog.rangeK[2][3] = fog.rangeK[2][1];
  put(out, offsetof(ViewUniform, fog), fog);
  std::array<Vec4<float>, MaxIndTexMtxs * 2> indTexMtxs;
  for (u32 i = 0; i < MaxIndTexMtxs; ++i) {
    const auto& mtx = g_gxState.indTexMtxs[i];
    indTexMtxs[i * 2] = Vec4{mtx.mtx.m0.x, mtx.mtx.m0.y, mtx.mtx.m1.x, mtx.mtx.m1.y};
    indTexMtxs[i * 2 + 1] = Vec4{mtx.mtx.m2.x, mtx.mtx.m2.y, std::exp2f(mtx.scaleExp), 0.0f};
  }
  put(out, offsetof(ViewUniform, indTexMtxs), indTexMtxs);
  put(out, offsetof(ViewUniform, depthRange),
      Vec4<float>{g_gxState.renderViewport.znear, g_gxState.renderViewport.zfar, 0.f, 0.f});
}

static void write_lights_uniform(u8* out) noexcept {
  put(out, offsetof(LightsUniform, lights), g_gxState.lights);
  std::array<u32, MaxColorChannels> lightMasks;
  for (u32 i = 0; i < MaxColorChannels; ++i) {
    lightMasks[i] = g_gxState.colorChannelState[i].lightMask.to_ulong();
  }
  put(out, offsetof(LightsUniform, lightMasks), lightMasks);
}

static void write_material_uniform(u8* out, const ShaderInfo& info) noexcept {
  put(out, offsetof(MaterialUniform, tevRegs), g_gxState.colorRegs);
  put(out, offsetof(MaterialUniform, kcolors), g_gxState.kcolors);
  std::array<Vec4<float>, MaxTexCoord> texCoordScales;
  for (u32 i = 0; i < MaxTexCoord; ++i) {
    const auto& scale = g_gxState.texCoordScales[i];
    texCoordScales[i] = Vec4{static_cast<f32>(scale.scaleS) + 1.0f, static_cast<f32>(scale.scaleT) + 1.0f, 0.0f, 0.0f};
  }
  put(out, offsetof(MaterialUniform, texCoordScales), texCoordScales);
  std::array<Vec4<float>, MaxTextures> textureSizeBias{};
  for (u32 i = 0; i < MaxTextures; ++i) {
    if (info.sampledTextures.test(i)) {
      textureSizeBias[i] = texture_size_bias(get_texture(static_cast<GXTexMapID>(i)));
    }
  }
  put(out, offsetof(MaterialUniform, textureSizeBias), textureSizeBias);
}

static gfx::Range push_uniform_piece(UniformPiece piece, const ShaderInfo& info) noexcept {
  if (piece >= UniformPieceCount) {
    return {};
  }
  gfx::Range range;
  u8* out = gfx::map_uniform(UniformPieceSizes[piece], range);
  if (out == nullptr) {
    return {};
  }
  switch (piece) {
  case UniformView:
    write_view_uniform(out, info.lineMode);
    break;
  case UniformPosMtx:
    for (u32 i = 0; i < MaxPnMtx; ++i) {
      put(out, i * sizeof(Mat3x4<float>), g_gxState.pnMtx[i].pos);
    }
    break;
  case UniformNrmMtx:
    for (u32 i = 0; i < MaxPnMtx; ++i) {
      put(out, i * sizeof(Mat3x4<float>), g_gxState.pnMtx[i].nrm);
    }
    break;
  case UniformLights:
    write_lights_uniform(out);
    break;
  case UniformMaterial:
    write_material_uniform(out, info);
    break;
  default:
    break;
  }
  return range;
}

void push_uniforms(const ShaderInfo& info, u8 pieces, std::array<u32, UniformPieceCount>& offsets) noexcept {
  ZoneScoped;
  for (u32 piece = 0; piece < UniformPieceCount; ++piece) {
    if ((pieces & 1 << piece) != 0) {
      offsets[piece] = push_uniform_piece(static_cast<UniformPiece>(piece), info).offset;
    }
  }
}
} // namespace aurora::gx
