// GLSL programs for a GX shader config, after gx/shader.cpp's WGSL in the form xost compiles; glsl.hpp has the
// bindings they read. Configs needing what xost doesn't compile yet are refused: matrices picked per vertex,
// lighting, fog, indirect texturing, and normal-based texgens (vulkan4xenon's ROADMAP.md, milestone 11).
#include "glsl.hpp"

#include "../gfx/hash.hpp"
#include "../gx/shader_info.hpp"
#include "../internal.hpp"

#include <dolphin/gx/GXEnum.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <string_view>

namespace aurora::gfx::vulkan4xenon::glsl {
using namespace std::string_literals;
using namespace std::string_view_literals;

namespace {
constexpr Module Log{"aurora::gfx::vulkan4xenon::glsl"};

constexpr size_t MaxInterpolants = 10;

struct Interpolant {
  std::string name; // out_<name> in the vertex program, in_<name> in the fragment program
  std::string_view type;
};

struct Plan {
  gx::ShaderInfo info;
  std::vector<Interpolant> interpolants;
};

constexpr bool is_texmtx_attr(u32 attr) noexcept { return attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX; }

bool refuse(std::string_view what) {
  Log.error("no GLSL yet for {}", what);
  return false;
}

bool make_plan(const gx::ShaderConfig& config, Plan& plan) {
  if (config.lineMode != 0) {
    return refuse("lines and points"sv);
  }
  if (config.tevStageCount == 0 || config.tevStageCount > gx::MaxTevStages) {
    Log.error("no GLSL for a config with {} TEV stages", config.tevStageCount);
    return false;
  }
  if (config.attrs[GX_VA_POS].attrType == GX_NONE) {
    return refuse("a config without positions"sv);
  }
  for (u32 attr = GX_VA_PNMTXIDX; attr <= GX_VA_TEX7; ++attr) {
    const auto& mapping = config.attrs[attr];
    if (mapping.attrType == GX_NONE) {
      continue;
    }
    if (attr == GX_VA_PNMTXIDX || is_texmtx_attr(attr)) {
      return refuse("matrices picked per vertex"sv);
    }
    const bool color = attr == GX_VA_CLR0 || attr == GX_VA_CLR1;
    if (mapping.compType > (color ? GX_RGBA8 : GX_F32)) {
      Log.error("no GLSL for vertex attribute {} of component type {}", attr, mapping.compType);
      return false;
    }
  }
  plan.info = gx::build_shader_info(config);
  const auto& info = plan.info;
  if (info.usesFog) {
    return refuse("fog"sv);
  }
  if (info.usedIndStages.any()) {
    return refuse("indirect texturing"sv);
  }
  for (u32 i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    if (stage.indTexMtxId != GX_ITM_OFF || stage.indTexWrapS != GX_ITW_OFF || stage.indTexWrapT != GX_ITW_OFF ||
        stage.indTexAddPrev || stage.channelId == GX_ALPHA_BUMP || stage.channelId == GX_ALPHA_BUMPN) {
      return refuse("indirect texturing"sv);
    }
  }
  for (u32 i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (info.sampledColorChannels.test(i) &&
        (config.colorChannels[i].lightingEnabled || config.colorChannels[i + GX_ALPHA0].lightingEnabled)) {
      return refuse("lighting"sv);
    }
  }
  for (u32 i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (info.sampledColorChannels.test(i)) {
      plan.interpolants.push_back({fmt::format("cc{}", i), "vec4"sv});
    }
  }
  for (u32 i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const auto& tcg = config.tcgs[i];
    if (tcg.type >= GX_TG_BUMP0 && tcg.type <= GX_TG_BUMP7) {
      return refuse("emboss texgens"sv);
    }
    if (tcg.src == GX_TG_BINRM || tcg.src == GX_TG_TANGENT) {
      return refuse("binormal and tangent texgens"sv);
    }
    if (tcg.type == GX_TG_MTX3x4) {
      plan.interpolants.push_back({fmt::format("tex{}_uvw", i), "vec3"sv});
    } else {
      plan.interpolants.push_back({fmt::format("tex{}_uv", i), "vec2"sv});
    }
  }
  if (plan.interpolants.size() > MaxInterpolants) {
    Log.error("no GLSL for a config with {} interpolants", plan.interpolants.size());
    return false;
  }
  return true;
}

// gx.hpp's uniform pieces as std140 blocks of vec4s, in UniformPiece order: each one's type, variable and members.
struct Piece {
  std::string_view type;
  std::string_view var;
  std::string members;
};

std::array<Piece, gx::UniformPieceCount> pieces() {
  return {{
      {"View"sv, "uview"sv,
       fmt::format("\n    vec4 viewport_sizes; // render, then logical"
                   "\n    vec4 line_params;"
                   "\n    vec4 proj[4];"
                   "\n    vec4 tex_mtx[{}];"
                   "\n    vec4 channel_colors[8]; // per channel: ambient, material, alpha ambient, alpha material"
                   "\n    vec4 postmtx[{}];"
                   "\n    vec4 fog[5];"
                   "\n    vec4 ind_mtx[{}];",
                   gx::MaxTexMtx * 3, gx::MaxPTTexMtx * 3, gx::MaxIndTexMtxs * 2)},
      {"PosMtx"sv, "upos"sv, fmt::format("\n    vec4 mtx[{}];", gx::MaxPnMtx * 3)},
      {"NrmMtx"sv, "unrm"sv, fmt::format("\n    vec4 mtx[{}];", gx::MaxPnMtx * 3)},
      {"Lights"sv, "ulights"sv, "\n    vec4 lights[40];\n    vec4 light_state;"s},
      {"Material"sv, "umat"sv,
       fmt::format("\n    vec4 tevreg[{}]; // TEVPREV, then TEVREG0-2"
                   "\n    vec4 kcolor[{}];"
                   "\n    vec4 texcoord_scale[{}];"
                   "\n    vec4 tex_size_bias[{}];",
                   gx::MaxTevRegs, gx::MaxKColors, gx::MaxTexCoord, gx::MaxTextures)},
  }};
}

// The blocks of the pieces a program reads, binding its UniformPiece, so the backend binds only what it reads.
std::string uniform_blocks(std::string_view program) {
  std::string blocks;
  const auto all = pieces();
  for (u32 i = 0; i < gx::UniformPieceCount; ++i) {
    if (program.find(fmt::format("{}.", all[i].var)) != std::string_view::npos) {
      blocks += fmt::format("layout(set = 0, binding = {}, std140) uniform {} {{{}\n}} {};\n", i, all[i].type,
                            all[i].members, all[i].var);
    }
  }
  return blocks;
}

// A 3x4 matrix of rows first to first + 2 in rows, times v, a vec4.
std::string mul3x4(std::string_view rows, u32 first, std::string_view v) {
  return fmt::format("vec3(dot({0}[{1}], {3}), dot({0}[{2}], {3}), dot({0}[{4}], {3}))", rows, first, first + 1, v,
                     first + 2);
}

std::string input_name(GXAttr attr) {
  if (attr == GX_VA_POS) {
    return "in_pos"s;
  }
  if (attr == GX_VA_NRM) {
    return "in_nrm"s;
  }
  if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
    return fmt::format("in_clr{}", attr - GX_VA_CLR0);
  }
  if (attr >= GX_VA_TEX0 && attr <= GX_VA_TEX7) {
    return fmt::format("in_tex{}_uv", attr - GX_VA_TEX0);
  }
  UNLIKELY FATAL("unhandled vtx attr {}", underlying(attr));
}

// The vertex program's value of a vertex attribute, noting the inputs it reads.
std::string vtx_attr(const gx::ShaderConfig& config, GXAttr attr, std::array<bool, gx::MaxVtxAttr>& inputs) {
  if (config.attrs[attr].attrType == GX_NONE) {
    if (attr == GX_VA_NRM) {
      return "vec3(1.0, 0.0, 0.0)"s;
    }
    if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
      return "vec4(0.0, 0.0, 0.0, 0.0)"s;
    }
    UNLIKELY FATAL("unmapped vtx attr {}", underlying(attr));
  }
  inputs[attr] = true;
  return input_name(attr);
}

bool vertex_body(const gx::ShaderConfig& config, const Plan& plan, std::array<bool, gx::MaxVtxAttr>& inputs,
                 std::string& body) {
  const auto& info = plan.info;
  body += fmt::format(
      "\n    vec4 pos4 = vec4({}, 1.0);"
      "\n    vec4 mv_pos4 = vec4({}, 1.0);"
      "\n    vec4 clip_pos = vec4(dot(uview.proj[0], mv_pos4), dot(uview.proj[1], mv_pos4),"
      " dot(uview.proj[2], mv_pos4), dot(uview.proj[3], mv_pos4));"
      "\n    gl_Position = clip_pos * vec4(1.0, -1.0, 1.0, 1.0);",
      vtx_attr(config, GX_VA_POS, inputs), mul3x4("upush.pos_mtx"sv, 0, "pos4"sv));

  // Unlit channels: the colour from one material source, the alpha from the alpha channel's.
  for (u32 i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (!info.sampledColorChannels.test(i)) {
      continue;
    }
    const auto material = [&](const gx::ColorChannelConfig& cc, u32 slot) {
      return cc.matSrc == GX_SRC_VTX ? vtx_attr(config, static_cast<GXAttr>(GX_VA_CLR0 + i), inputs)
                                     : fmt::format("uview.channel_colors[{}]", i * 4 + slot);
    };
    body += fmt::format("\n    vec4 cc{0} = vec4({1}.rgb, {2}.a);\n    out_cc{0} = cc{0};", i,
                        material(config.colorChannels[i], 1), material(config.colorChannels[i + GX_ALPHA0], 3));
  }

  for (u32 i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const auto& tcg = config.tcgs[i];
    std::string source;
    if (tcg.src >= GX_TG_TEX0 && tcg.src <= GX_TG_TEX7) {
      source = fmt::format("vec4({}, 1.0, 1.0)", vtx_attr(config, GXAttr(GX_VA_TEX0 + (tcg.src - GX_TG_TEX0)), inputs));
    } else if (tcg.src == GX_TG_POS) {
      source = fmt::format("vec4({}, 1.0)", vtx_attr(config, GX_VA_POS, inputs));
    } else if (tcg.src == GX_TG_NRM) {
      source = fmt::format("vec4({}, 1.0)", vtx_attr(config, GX_VA_NRM, inputs));
    } else if (tcg.src == GX_TG_COLOR0 || tcg.src == GX_TG_COLOR1) {
      // A colour-source texgen takes the channel's rasterized output, as gx/shader.cpp does.
      const u32 ch = tcg.src == GX_TG_COLOR0 ? 0 : 1;
      source = info.sampledColorChannels.test(ch) ? fmt::format("cc{}", ch)
                                                  : vtx_attr(config, static_cast<GXAttr>(GX_VA_CLR0 + ch), inputs);
    } else {
      Log.error("no GLSL for texgen source {}", underlying(tcg.src));
      return false;
    }
    body += fmt::format("\n    vec4 tc{} = {};", i, source);
    const auto tc = fmt::format("tc{}", i);
    std::string value;
    if (tcg.type == GX_TG_MTX2x4 || tcg.type == GX_TG_MTX3x4) {
      if (tcg.mtx == GX_IDENTITY) {
        value = fmt::format("{}.xyz", tc);
      } else if (tcg.mtx >= GX_TEXMTX0) {
        value = mul3x4("uview.tex_mtx"sv, (tcg.mtx - GX_TEXMTX0) / 3 * 3, tc);
      } else {
        value = mul3x4("upos.mtx"sv, tcg.mtx / 3 * 3, tc);
      }
      if (tcg.type == GX_TG_MTX2x4) {
        body += fmt::format("\n    vec3 tc{0}_mtx = {1};", i, value);
        value = fmt::format("vec3(tc{}_mtx.xy, 1.0)", i);
      }
    } else if (tcg.type == GX_TG_SRTG) {
      value = fmt::format("vec3({}.xy, 1.0)", tc);
    } else {
      Log.error("no GLSL for texgen type {}", underlying(tcg.type));
      return false;
    }
    body += fmt::format("\n    vec3 tc{}_tmp = {};", i, value);
    std::string tmp = fmt::format("tc{}_tmp", i);
    if (tcg.normalize) {
      body += fmt::format("\n    vec3 tc{0}_norm = normalize({1});", i, tmp);
      tmp = fmt::format("tc{}_norm", i);
    }
    std::string projected = tmp;
    if (tcg.postMtx != GX_PTIDENTITY) {
      body += fmt::format("\n    vec4 tc{0}_post = vec4({1}, 1.0);\n    vec3 tc{0}_proj = {2};", i, tmp,
                          mul3x4("uview.postmtx"sv, (tcg.postMtx - GX_PTTEXMTX0) / 3 * 3, fmt::format("tc{}_post", i)));
      projected = fmt::format("tc{}_proj", i);
    }
    if (tcg.type == GX_TG_MTX3x4) {
      body += fmt::format("\n    out_tex{}_uvw = {};", i, projected);
    } else {
      body += fmt::format("\n    out_tex{}_uv = {}.xy;", i, projected);
    }
  }
  return true;
}

// Attribute locations as xost gives them usages: position 0, colour 0 1, texture coordinates from 2 (resolve.py), and
// the rest after them, in decode_vertices' order.
VertexLayout make_layout(const std::array<bool, gx::MaxVtxAttr>& inputs) {
  VertexLayout layout;
  const auto add = [&](GXAttr attr, uint32_t location, AttrFormat format, uint32_t size) {
    layout.attributes.push_back({location, format, layout.stride, attr});
    layout.stride += size;
  };
  add(GX_VA_POS, 0, AttrFormat::F32x3, 12);
  if (inputs[GX_VA_CLR0]) {
    add(GX_VA_CLR0, 1, AttrFormat::U8x4, 4);
  }
  uint32_t location = 2;
  for (u32 attr = GX_VA_TEX0; attr <= GX_VA_TEX7; ++attr) {
    if (inputs[attr]) {
      add(static_cast<GXAttr>(attr), location++, AttrFormat::F32x2, 8);
    }
  }
  if (inputs[GX_VA_NRM]) {
    add(GX_VA_NRM, location++, AttrFormat::F32x3, 12);
  }
  if (inputs[GX_VA_CLR1]) {
    add(GX_VA_CLR1, location++, AttrFormat::U8x4, 4);
  }
  return layout;
}

std::string_view input_type(AttrFormat format) {
  switch (format) {
  case AttrFormat::F32x1:
    return "float"sv;
  case AttrFormat::F32x2:
    return "vec2"sv;
  case AttrFormat::F32x3:
    return "vec3"sv;
  case AttrFormat::U8x4:
    return "vec4"sv;
  }
  return {};
}

std::string vertex_source(const Plan& plan, const VertexLayout& layout, const std::string& body) {
  std::string decls = "layout(push_constant, std140) uniform Push {\n    vec4 pos_mtx[3];\n} upush;\n";
  for (const auto& attribute : layout.attributes) {
    decls += fmt::format("layout(location = {}) in {} {};\n", attribute.location, input_type(attribute.format),
                         input_name(attribute.attr));
  }
  for (size_t i = 0; i < plan.interpolants.size(); ++i) {
    decls +=
        fmt::format("layout(location = {}) out {} out_{};\n", i, plan.interpolants[i].type, plan.interpolants[i].name);
  }
  const auto program = fmt::format("{}void main() {{{}\n}}\n", decls, body);
  return fmt::format("#version 450\n{}{}", uniform_blocks(program), program);
}

std::string_view chan_comp(GXTevColorChan chan) noexcept {
  switch (chan) {
  case GX_CH_RED:
    return "r";
  case GX_CH_GREEN:
    return "g";
  case GX_CH_BLUE:
    return "b";
  case GX_CH_ALPHA:
    return "a";
  default:
    return "?";
  }
}

bool uses_texture_sample(const gx::TevStage& stage) noexcept {
  if (stage.texMapId == GX_TEXMAP_NULL) {
    return false;
  }
  const auto& c = stage.colorPass;
  const auto& a = stage.alphaPass;
  return c.a == GX_CC_TEXC || c.a == GX_CC_TEXA || c.b == GX_CC_TEXC || c.b == GX_CC_TEXA || c.c == GX_CC_TEXC ||
         c.c == GX_CC_TEXA || c.d == GX_CC_TEXC || c.d == GX_CC_TEXA || a.a == GX_CA_TEXA || a.b == GX_CA_TEXA ||
         a.c == GX_CA_TEXA || a.d == GX_CA_TEXA;
}

// A TEV register as the program holds it at some stage: the locals with its colour and its alpha, and whether each is
// clamped to [0, 1] (normalized) or may need tev_overflow's wrap.
struct Reg {
  std::string color;
  std::string alpha;
  bool colorNormalized = true;
  bool alphaNormalized = true;
};

std::string konst_color(GXTevKColorSel sel) {
  if (sel >= GX_TEV_KCSEL_1 && sel <= GX_TEV_KCSEL_1_8) {
    return fmt::format("vec3({}.0 / 8.0)", 8 - (sel - GX_TEV_KCSEL_1));
  }
  if (sel >= GX_TEV_KCSEL_K0 && sel <= GX_TEV_KCSEL_K3) {
    return fmt::format("umat.kcolor[{}].rgb", sel - GX_TEV_KCSEL_K0);
  }
  static constexpr std::array comps{"r"sv, "g"sv, "b"sv, "a"sv};
  if (sel >= GX_TEV_KCSEL_K0_R && sel <= GX_TEV_KCSEL_K3_A) {
    const u32 k = (sel - GX_TEV_KCSEL_K0_R) % 4;
    const u32 comp = (sel - GX_TEV_KCSEL_K0_R) / 4;
    return fmt::format("vec3(umat.kcolor[{}].{})", k, comps[comp]);
  }
  FATAL("invalid kcSel {}", underlying(sel));
}

std::string konst_alpha(GXTevKAlphaSel sel) {
  if (sel >= GX_TEV_KASEL_1 && sel <= GX_TEV_KASEL_1_8) {
    return fmt::format("({}.0 / 8.0)", 8 - (sel - GX_TEV_KASEL_1));
  }
  static constexpr std::array comps{"r"sv, "g"sv, "b"sv, "a"sv};
  if (sel >= GX_TEV_KASEL_K0_R && sel <= GX_TEV_KASEL_K3_A) {
    const u32 k = (sel - GX_TEV_KASEL_K0_R) % 4;
    const u32 comp = (sel - GX_TEV_KASEL_K0_R) / 4;
    return fmt::format("umat.kcolor[{}].{}", k, comps[comp]);
  }
  FATAL("invalid kaSel {}", underlying(sel));
}

// The program's fragment body for its TEV stages, and the final colour's locals.
class Fragment {
public:
  Fragment(const gx::ShaderConfig& config, const gx::ShaderInfo& info) : m_config(config) {
    static constexpr std::array zero{"vec3(0.0)"sv, "0.0"sv};
    for (u32 i = 0; i < gx::MaxTevRegs; ++i) {
      if (info.loadsTevReg.test(i)) {
        m_regs[i] = {fmt::format("umat.tevreg[{}].rgb", i), fmt::format("umat.tevreg[{}].a", i), true, true};
      } else {
        m_regs[i] = {std::string{zero[0]}, std::string{zero[1]}, true, true};
      }
    }
  }

  // Each stage's colour and alpha into a local of their own, as the stage's output register then holds them.
  void stages() {
    for (u32 idx = 0; idx < m_config.tevStageCount; ++idx) {
      const auto& stage = m_config.tevStages[idx];
      const auto& c = stage.colorPass;
      const std::string color = color_op(idx, stage, color_arg(idx, stage, c.a, 'a'), color_arg(idx, stage, c.b, 'b'),
                                         color_arg(idx, stage, c.c, 'c'), color_value(idx, stage, c.d));
      const auto& a = stage.alphaPass;
      const std::string alpha = alpha_op(idx, stage, alpha_arg(idx, stage, a.a, 'a'), alpha_arg(idx, stage, a.b, 'b'),
                                         alpha_arg(idx, stage, a.c, 'c'), alpha_value(idx, stage, a.d));
      m_body += fmt::format("\n    // TEV stage {0}\n    vec3 s{0}_color = {1};\n    float s{0}_alpha = {2};", idx,
                            color, alpha);
      Reg& colorReg = m_regs[stage.colorOp.outReg];
      colorReg.color = fmt::format("s{}_color", idx);
      colorReg.colorNormalized = stage.colorOp.clamp;
      Reg& alphaReg = m_regs[stage.alphaOp.outReg];
      alphaReg.alpha = fmt::format("s{}_alpha", idx);
      alphaReg.alphaNormalized = stage.alphaOp.clamp;
    }
  }

  // The output colour, in [0, 1] where GX wraps an unclamped result, and the alpha compare as one test.
  bool finish() {
    const auto& last = m_config.tevStages[m_config.tevStageCount - 1];
    const Reg& colorReg = m_regs[last.colorOp.outReg];
    const Reg& alphaReg = m_regs[last.alphaOp.outReg];
    std::string color = colorReg.color;
    std::string alpha = alphaReg.alpha;
    if (!colorReg.colorNormalized) {
      m_body += fmt::format("\n    vec3 out_color = {};", overflow(color));
      color = "out_color"s;
    }
    if (!alphaReg.alphaNormalized) {
      m_body += fmt::format("\n    float out_alpha = {};", overflow(alpha));
      alpha = "out_alpha"s;
    }
    if (m_config.alphaCompare && !alpha_compare(alpha, alphaReg.alphaNormalized)) {
      return false;
    }
    m_body += fmt::format("\n    o_color = vec4({}, {});", color, alpha);
    return true;
  }

  const std::string& body() const noexcept { return m_body; }

private:
  // tev_overflow, inline: the value modulo 256/255 in the byte domain, as GX wraps an unclamped register.
  static std::string overflow(std::string_view v) {
    return fmt::format("(({0}) * 255.0 - floor(({0}) * 255.0 / 256.0) * 256.0) / 255.0", v);
  }

  std::string sampled(u32 idx, const gx::TevStage& stage) const {
    CHECK(stage.texMapId >= GX_TEXMAP0 && stage.texMapId <= GX_TEXMAP7, "invalid texture {} for stage {}",
          underlying(stage.texMapId), idx);
    return fmt::format("sampled{}", idx);
  }

  std::string color_value(u32 idx, const gx::TevStage& stage, GXTevColorArg arg) const {
    switch (arg) {
      DEFAULT_FATAL("invalid color arg {}", underlying(arg));
    case GX_CC_CPREV:
    case GX_CC_C0:
    case GX_CC_C1:
    case GX_CC_C2:
      return m_regs[(arg - GX_CC_CPREV) / 2].color;
    case GX_CC_APREV:
    case GX_CC_A0:
    case GX_CC_A1:
    case GX_CC_A2:
      return fmt::format("vec3({})", m_regs[(arg - GX_CC_APREV) / 2].alpha);
    case GX_CC_TEXC: {
      if (stage.texMapId == GX_TEXMAP_NULL) {
        return "vec3(1.0)"s;
      }
      const auto& swap = m_config.tevSwapTable[stage.tevSwapTex];
      return fmt::format("{}.{}{}{}", sampled(idx, stage), chan_comp(swap.red), chan_comp(swap.green),
                         chan_comp(swap.blue));
    }
    case GX_CC_TEXA: {
      if (stage.texMapId == GX_TEXMAP_NULL) {
        return "vec3(1.0)"s;
      }
      const auto& swap = m_config.tevSwapTable[stage.tevSwapTex];
      return fmt::format("vec3({}.{})", sampled(idx, stage), chan_comp(swap.alpha));
    }
    case GX_CC_RASC: {
      if (stage.channelId == GX_COLOR_ZERO || stage.channelId == GX_COLOR_NULL) {
        return "vec3(0.0)"s;
      }
      const auto& swap = m_config.tevSwapTable[stage.tevSwapRas];
      return fmt::format("in_cc{}.{}{}{}", gx::color_channel(stage.channelId), chan_comp(swap.red),
                         chan_comp(swap.green), chan_comp(swap.blue));
    }
    case GX_CC_RASA: {
      if (stage.channelId == GX_COLOR_ZERO || stage.channelId == GX_COLOR_NULL) {
        return "vec3(0.0)"s;
      }
      const auto& swap = m_config.tevSwapTable[stage.tevSwapRas];
      return fmt::format("vec3(in_cc{}.{})", gx::color_channel(stage.channelId), chan_comp(swap.alpha));
    }
    case GX_CC_ONE:
      return "vec3(1.0)"s;
    case GX_CC_HALF:
      return "vec3(0.5)"s;
    case GX_CC_KONST:
      return konst_color(stage.kcSel);
    case GX_CC_ZERO:
      return "vec3(0.0)"s;
    }
  }

  std::string alpha_value(u32 idx, const gx::TevStage& stage, GXTevAlphaArg arg) const {
    switch (arg) {
      DEFAULT_FATAL("invalid alpha arg {}", underlying(arg));
    case GX_CA_APREV:
    case GX_CA_A0:
    case GX_CA_A1:
    case GX_CA_A2:
      return m_regs[arg - GX_CA_APREV].alpha;
    case GX_CA_TEXA: {
      if (stage.texMapId == GX_TEXMAP_NULL) {
        return "1.0"s;
      }
      const auto& swap = m_config.tevSwapTable[stage.tevSwapTex];
      return fmt::format("{}.{}", sampled(idx, stage), chan_comp(swap.alpha));
    }
    case GX_CA_RASA: {
      if (stage.channelId == GX_COLOR_ZERO || stage.channelId == GX_COLOR_NULL) {
        return "0.0"s;
      }
      const auto& swap = m_config.tevSwapTable[stage.tevSwapRas];
      return fmt::format("in_cc{}.{}", gx::color_channel(stage.channelId), chan_comp(swap.alpha));
    }
    case GX_CA_KONST:
      return konst_alpha(stage.kaSel);
    case GX_CA_ZERO:
      return "0.0"s;
    }
  }

  bool color_normalized(GXTevColorArg arg) const {
    if (arg >= GX_CC_CPREV && arg <= GX_CC_A2) {
      const Reg& reg = m_regs[(arg - GX_CC_CPREV) / 2];
      return (arg - GX_CC_CPREV) % 2 == 0 ? reg.colorNormalized : reg.alphaNormalized;
    }
    return true;
  }

  bool alpha_normalized(GXTevAlphaArg arg) const { return arg > GX_CA_A2 || m_regs[arg - GX_CA_APREV].alphaNormalized; }

  // Arguments a, b and c: a register not clamped since it was written wraps, through a local of its own.
  std::string color_arg(u32 idx, const gx::TevStage& stage, GXTevColorArg arg, char which) {
    const auto value = color_value(idx, stage, arg);
    if (color_normalized(arg)) {
      return fmt::format("({})", value);
    }
    m_body += fmt::format("\n    vec3 s{}_color_{} = {};", idx, which, overflow(value));
    return fmt::format("s{}_color_{}", idx, which);
  }

  std::string alpha_arg(u32 idx, const gx::TevStage& stage, GXTevAlphaArg arg, char which) {
    const auto value = alpha_value(idx, stage, arg);
    if (alpha_normalized(arg)) {
      return fmt::format("({})", value);
    }
    m_body += fmt::format("\n    float s{}_alpha_{} = {};", idx, which, overflow(value));
    return fmt::format("s{}_alpha_{}", idx, which);
  }

  static std::string_view bias(GXTevBias bias) {
    switch (bias) {
      DEFAULT_FATAL("invalid tev bias {}", underlying(bias));
    case GX_TB_ZERO:
      return ""sv;
    case GX_TB_ADDHALF:
      return " + 0.5"sv;
    case GX_TB_SUBHALF:
      return " - 0.5"sv;
    }
  }

  static std::string_view scale(GXTevScale scale) {
    switch (scale) {
      DEFAULT_FATAL("invalid tev scale {}", underlying(scale));
    case GX_CS_SCALE_1:
      return ""sv;
    case GX_CS_SCALE_2:
      return " * 2.0"sv;
    case GX_CS_SCALE_4:
      return " * 4.0"sv;
    case GX_CS_DIVIDE_2:
      return " / 2.0"sv;
    }
  }

  // The 8-, 16- and 24-bit keys GX compares, rounded as the WGSL rounds them. color selects the per-channel RGB8 key.
  static std::string compare_key(GXTevOp op, std::string_view v, bool scalar) {
    switch (op) {
    case GX_TEV_COMP_R8_GT:
    case GX_TEV_COMP_R8_EQ:
      return scalar ? fmt::format("floor({} * 255.0 + 0.5)", v) : fmt::format("floor({}.r * 255.0 + 0.5)", v);
    case GX_TEV_COMP_GR16_GT:
    case GX_TEV_COMP_GR16_EQ:
      // A scalar's .rg reads as the scalar twice: 255 * (1 + 256).
      return scalar ? fmt::format("floor({} * 65535.0 + 0.5)", v)
                    : fmt::format("floor(dot({}.rg * 255.0, vec2(1.0, 256.0)) + 0.5)", v);
    case GX_TEV_COMP_BGR24_GT:
    case GX_TEV_COMP_BGR24_EQ:
      return scalar ? fmt::format("floor({} * 16777215.0 + 0.5)", v)
                    : fmt::format("floor(dot({}.rgb * 255.0, vec3(1.0, 256.0, 65536.0)) + 0.5)", v);
    default:
      return scalar ? fmt::format("floor({} * 255.0 + 0.5)", v) : fmt::format("floor({} * 255.0 + 0.5)", v);
    }
  }

  static bool is_greater(GXTevOp op) {
    return op == GX_TEV_COMP_R8_GT || op == GX_TEV_COMP_GR16_GT || op == GX_TEV_COMP_BGR24_GT ||
           op == GX_TEV_COMP_RGB8_GT;
  }

  // The op before its clamp. A compare picks c or 0 by a factor of 1 or 0 (step), which xost compiles where it
  // doesn't compile the WGSL's select (xost: a branch writes one variable once). The keys are whole numbers, so 0.5
  // separates them.
  static std::string op_expr(const gx::TevOp& tevOp, std::string_view a, std::string_view b, std::string_view c,
                             std::string_view d, bool scalar) {
    switch (tevOp.op) {
    case GX_TEV_ADD:
      return fmt::format("((mix({}, {}, {}) + {}){}){}", a, b, c, d, bias(tevOp.bias), scale(tevOp.scale));
    case GX_TEV_SUB:
      return fmt::format("(({} - mix({}, {}, {})){}){}", d, a, b, c, bias(tevOp.bias), scale(tevOp.scale));
    default: {
      // RGB8's per-channel compare gives a vector factor; the rest one factor for every channel.
      const bool vector = !scalar && (tevOp.op == GX_TEV_COMP_RGB8_GT || tevOp.op == GX_TEV_COMP_RGB8_EQ);
      const auto half = vector ? "vec3(0.5)"sv : "0.5"sv;
      const auto one = vector ? "vec3(1.0)"sv : "1.0"sv;
      const auto ka = compare_key(tevOp.op, a, scalar);
      const auto kb = compare_key(tevOp.op, b, scalar);
      const auto factor = is_greater(tevOp.op) ? fmt::format("step({}, {} - {})", half, ka, kb)
                                               : fmt::format("({} - step({}, abs({} - {})))", one, half, ka, kb);
      return fmt::format("({} * {} + {})", c, factor, d);
    }
    }
  }

  static std::string color_op(u32, const gx::TevStage& stage, std::string_view a, std::string_view b,
                              std::string_view c, std::string_view d) {
    const auto expr = op_expr(stage.colorOp, a, b, c, d, false);
    return stage.colorOp.clamp ? fmt::format("clamp({}, 0.0, 1.0)", expr) : fmt::format("clamp({}, -4.0, 4.0)", expr);
  }

  static std::string alpha_op(u32, const gx::TevStage& stage, std::string_view a, std::string_view b,
                              std::string_view c, std::string_view d) {
    const auto expr = op_expr(stage.alphaOp, a, b, c, d, true);
    return stage.alphaOp.clamp ? fmt::format("clamp({}, 0.0, 1.0)", expr) : fmt::format("clamp({}, -4.0, 4.0)", expr);
  }

  // One compare of round(alpha * 255) with ref: 1 or 0 when it is known, else the expression, 1 where it passes.
  struct Pass {
    int constant = -1;
    std::string expr;
  };

  static Pass compare(GXCompare comp, u32 ref) {
    const auto pass = [](std::string expr) { return Pass{-1, std::move(expr)}; };
    switch (comp) {
      DEFAULT_FATAL("invalid alpha comp {}", underlying(comp));
    case GX_NEVER:
      return {0, {}};
    case GX_LESS:
      return ref == 0 ? Pass{0, {}} : pass(fmt::format("step(0.5, {}.0 - alpha_byte)", ref));
    case GX_LEQUAL:
      return ref >= 255 ? Pass{1, {}} : pass(fmt::format("step(-0.5, {}.0 - alpha_byte)", ref));
    case GX_EQUAL:
      return pass(fmt::format("(1.0 - step(0.5, abs(alpha_byte - {}.0)))", ref));
    case GX_NEQUAL:
      return pass(fmt::format("step(0.5, abs(alpha_byte - {}.0))", ref));
    case GX_GEQUAL:
      return ref == 0 ? Pass{1, {}} : pass(fmt::format("step(-0.5, alpha_byte - {}.0)", ref));
    case GX_GREATER:
      return ref >= 255 ? Pass{0, {}} : pass(fmt::format("step(0.5, alpha_byte - {}.0)", ref));
    case GX_ALWAYS:
      return {1, {}};
    }
  }

  static Pass combine(GXAlphaOp op, const Pass& lhs, const Pass& rhs) {
    const auto value = [](const Pass& p) { return p.constant == -1 ? p.expr : fmt::format("{}.0", p.constant); };
    if (lhs.constant != -1 && rhs.constant != -1) {
      switch (op) {
      case GX_AOP_AND:
        return {lhs.constant & rhs.constant, {}};
      case GX_AOP_OR:
        return {lhs.constant | rhs.constant, {}};
      case GX_AOP_XOR:
        return {lhs.constant ^ rhs.constant, {}};
      case GX_AOP_XNOR:
        return {1 - (lhs.constant ^ rhs.constant), {}};
      default:
        FATAL("invalid alpha compare op {}", underlying(op));
      }
    }
    switch (op) {
      DEFAULT_FATAL("invalid alpha compare op {}", underlying(op));
    case GX_AOP_AND:
      if (lhs.constant == 0 || rhs.constant == 0) {
        return {0, {}};
      }
      if (lhs.constant == 1 || lhs.expr == rhs.expr) {
        return rhs;
      }
      return rhs.constant == 1 ? lhs : Pass{-1, fmt::format("({} * {})", lhs.expr, rhs.expr)};
    case GX_AOP_OR:
      if (lhs.constant == 1 || rhs.constant == 1) {
        return {1, {}};
      }
      if (lhs.constant == 0 || lhs.expr == rhs.expr) {
        return rhs;
      }
      return rhs.constant == 0 ? lhs : Pass{-1, fmt::format("max({}, {})", lhs.expr, rhs.expr)};
    case GX_AOP_XOR:
      return {-1, fmt::format("abs({} - {})", value(lhs), value(rhs))};
    case GX_AOP_XNOR:
      return {-1, fmt::format("(1.0 - abs({} - {}))", value(lhs), value(rhs))};
    }
  }

  // GX keeps a fragment when round(alpha * 255) passes; one test of the combined pass factor discards the rest. An
  // alpha its stage didn't clamp is clamped first.
  bool alpha_compare(const std::string& alpha, bool normalized) {
    const auto& ac = m_config.alphaCompare;
    const Pass pass = combine(ac.op, compare(ac.comp0, ac.ref0), compare(ac.comp1, ac.ref1));
    if (pass.constant == 1) {
      return true;
    }
    if (pass.constant == 0) {
      // Nothing is drawn; xost compiles no program that only discards, and the backend skips such a draw.
      return refuse("an alpha compare that discards every fragment"sv);
    }
    m_body += fmt::format(
        "\n    // Alpha compare"
        "\n    float alpha_byte = floor({} * 255.0 + 0.5);"
        "\n    float alpha_pass = {};"
        "\n    if (alpha_pass < 0.5) {{\n        discard;\n    }}",
        normalized ? alpha : fmt::format("clamp({}, 0.0, 1.0)", alpha), pass.expr);
    return true;
  }

  const gx::ShaderConfig& m_config;
  std::array<Reg, gx::MaxTevRegs> m_regs;
  std::string m_body;
};

bool fragment_source(const gx::ShaderConfig& config, const Plan& plan, std::string& source) {
  const auto& info = plan.info;
  std::string pre;
  for (u32 i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    if (config.tcgs[i].type == GX_TG_MTX3x4) {
      pre += fmt::format("\n    vec2 tex{0}_uv = in_tex{0}_uvw.xy / in_tex{0}_uvw.z;", i);
    } else {
      pre += fmt::format("\n    vec2 tex{0}_uv = in_tex{0}_uv;", i);
    }
  }
  for (u32 i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    if (!uses_texture_sample(stage)) {
      continue;
    }
    if (stage.texCoordId == GX_TEXCOORD_NULL) {
      Log.error("no GLSL for TEV stage {}, which samples a texture with no coordinates", i);
      return false;
    }
    // xost takes only a bias known when it compiles, so the material's LOD bias goes in the sampler.
    pre += fmt::format("\n    vec4 sampled{0} = texture(tex{1}, tex{2}_uv);", i, underlying(stage.texMapId),
                       underlying(stage.texCoordId));
  }
  Fragment fragment(config, info);
  fragment.stages();
  if (!fragment.finish()) {
    return false;
  }

  std::string decls;
  for (u32 i = 0; i < info.sampledTextures.size(); ++i) {
    if (info.sampledTextures.test(i)) {
      decls += fmt::format("layout(set = 1, binding = {0}) uniform sampler2D tex{0};\n", i);
    }
  }
  for (size_t i = 0; i < plan.interpolants.size(); ++i) {
    decls +=
        fmt::format("layout(location = {}) in {} in_{};\n", i, plan.interpolants[i].type, plan.interpolants[i].name);
  }
  decls += "layout(location = 0) out vec4 o_color;\n";
  const auto program = fmt::format("{}void main() {{{}{}\n}}\n", decls, pre, fragment.body());
  source = fmt::format("#version 450\n{}{}", uniform_blocks(program), program);
  return true;
}
} // namespace

Programs build_programs(const gx::ShaderConfig& config) {
  Plan plan;
  Programs programs;
  std::array<bool, gx::MaxVtxAttr> inputs{};
  std::string body;
  if (!make_plan(config, plan) || !vertex_body(config, plan, inputs, body) ||
      !fragment_source(config, plan, programs.fragment)) {
    return {};
  }
  programs.layout = make_layout(inputs);
  programs.vertex = vertex_source(plan, programs.layout, body);
  return programs;
}
} // namespace aurora::gfx::vulkan4xenon::glsl

namespace aurora::gfx::vulkan4xenon::glsl {
namespace {
// What a copy to each format writes, from the sampled colour c and its intensity: tex_copy_conv.cpp's conversions.
struct CopyConversion {
  GXTexFmt format;
  std::string_view result;
};
constexpr std::array CopyConversions{
    CopyConversion{GX_TF_RGBA8, "c"sv},
    CopyConversion{GX_TF_I4, "vec4(floor(intensity * 16.0) / 15.0)"sv},
    CopyConversion{GX_TF_I8, "vec4(intensity)"sv},
    CopyConversion{GX_TF_IA4, "vec4(vec3(floor(intensity * 16.0) / 15.0), floor(c.a * 16.0) / 15.0)"sv},
    CopyConversion{GX_TF_IA8, "vec4(vec3(intensity), c.a)"sv},
    CopyConversion{GX_TF_RGB565, "vec4(c.rgb, 1.0)"sv},
    CopyConversion{GX_CTF_R4, "vec4(floor(c.r * 16.0) / 15.0)"sv},
    CopyConversion{GX_CTF_RA4, "vec4(vec3(floor(c.r * 16.0) / 15.0), floor(c.a * 16.0) / 15.0)"sv},
    CopyConversion{GX_CTF_RA8, "c.rrra"sv},
    CopyConversion{GX_CTF_A8, "c.aaaa"sv},
    CopyConversion{GX_CTF_R8, "c.rrrr"sv},
    CopyConversion{GX_CTF_G8, "c.gggg"sv},
    CopyConversion{GX_CTF_B8, "c.bbbb"sv},
    CopyConversion{GX_CTF_RG8, "c.rrrg"sv},
    CopyConversion{GX_CTF_GB8, "c.gggb"sv},
};
} // namespace

std::string clear_vertex() {
  return "#version 450\n"
         "layout(location = 0) in vec2 in_position;\n"
         "void main() {\n"
         "    gl_Position = vec4(in_position, 0.0, 1.0);\n"
         "}\n";
}

std::string clear_fragment() {
  return "#version 450\n"
         "layout(push_constant, std140) uniform Push {\n"
         "    vec4 color;\n"
         "} upush;\n"
         "layout(location = 0) out vec4 o_color;\n"
         "void main() {\n"
         "    o_color = upush.color;\n"
         "}\n";
}

// uv_transform is the copy's source rectangle, as offset and scale (recording.cpp, resolve_pass_into).
std::string copy_vertex() {
  return "#version 450\n"
         "layout(push_constant, std140) uniform Push {\n"
         "    vec4 uv_transform;\n"
         "} upush;\n"
         "layout(location = 0) in vec2 in_position;\n"
         "layout(location = 1) in vec2 in_uv;\n"
         "layout(location = 0) out vec2 out_uv;\n"
         "void main() {\n"
         "    gl_Position = vec4(in_position, 0.0, 1.0);\n"
         "    out_uv = in_uv * upush.uv_transform.zw + upush.uv_transform.xy;\n"
         "}\n";
}

// Sampled alpha is read as 1 in a copy from an EFB without alpha (tex_copy_conv.cpp, sample_efb).
std::string copy_fragment(GXTexFmt format, bool opaqueAlpha) {
  const auto conversion = std::ranges::find(CopyConversions, format, &CopyConversion::format);
  if (conversion == CopyConversions.end()) {
    return {};
  }
  const bool intensity = conversion->result.find("intensity") != std::string_view::npos;
  return fmt::format(
      "#version 450\n"
      "layout(set = 0, binding = 0) uniform sampler2D source;\n"
      "layout(location = 0) in vec2 in_uv;\n"
      "layout(location = 0) out vec4 o_color;\n"
      "void main() {{\n"
      "    vec4 sampled = texture(source, in_uv);\n"
      "    vec4 c = {};{}\n"
      "    o_color = {};\n"
      "}}\n",
      opaqueAlpha ? "vec4(sampled.rgb, 1.0)"sv : "sampled"sv,
      intensity ? "\n    float intensity = dot(c.rgb, vec3(0.257, 0.504, 0.098)) + 16.0 / 255.0;"sv : ""sv,
      conversion->result);
}

bool converts(GXTexFmt format) noexcept {
  return format != GX_TF_RGBA8 &&
         std::ranges::find(CopyConversions, format, &CopyConversion::format) != CopyConversions.end();
}

std::vector<std::pair<std::string, bool>> fixed_programs() {
  std::vector<std::pair<std::string, bool>> programs{
      {clear_vertex(), false},
      {clear_fragment(), true},
      {copy_vertex(), false},
  };
  for (const auto& conversion : CopyConversions) {
    for (const bool opaqueAlpha : {false, true}) {
      programs.emplace_back(copy_fragment(conversion.format, opaqueAlpha), true);
    }
  }
  return programs;
}

uint64_t program_hash(std::string_view source, bool fragment) noexcept {
  return XXH3_64bits_withSeed(source.data(), source.size(), fragment ? 1 : 0);
}
} // namespace aurora::gfx::vulkan4xenon::glsl
