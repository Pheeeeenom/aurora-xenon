// GLSL programs for a GX shader config, after gx/shader.cpp's WGSL in the form xost compiles; glsl.hpp has the
// bindings they read. Configs needing what these programs don't express yet are refused: lines and points, emboss,
// binormal and tangent texgens, and more interpolants than xost gives a program (vulkan4xenon's ROADMAP.md, milestone
// 11).
#include "glsl.hpp"

#include "../gfx/hash.hpp"
#include "../gx/shader_info.hpp"
#include "../internal.hpp"

#include <dolphin/gx/GXEnum.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <mutex>
#include <string_view>
#include <unordered_set>

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

// What a config needs that these programs don't express yet, each with what a generator for it would need to know,
// in the order make_plan finds them.
using Needs = std::vector<std::string>;

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

std::string join(const std::vector<std::string>& items, std::string_view separator) {
  std::string joined;
  for (const auto& item : items) {
    if (!joined.empty()) {
      joined += separator;
    }
    joined += item;
  }
  return joined;
}

std::mutex s_refusedLock;
std::unordered_set<uint64_t> s_refused;

// A refused config's needs, once per config under its hash, as the pipeline cache asks for a config once for each
// pipeline it's in. The hash with every channel's lights cleared groups the configs that differ only in them.
void refuse(const gx::ShaderConfig& config, const Needs& needs) {
  const uint64_t hash = XXH3_64bits(&config, sizeof(config));
  {
    std::lock_guard lock(s_refusedLock);
    if (!s_refused.insert(hash).second) {
      return;
    }
  }
  gx::ShaderConfig unlit = config;
  for (auto& channel : unlit.colorChannels) {
    channel.lightMask = 0;
    channel.unitSpotLights = 0;
    channel.unitDistanceLights = 0;
  }
  Log.error("no GLSL yet for config {:016x} ({:016x} without its lights): {}", hash, XXH3_64bits(&unlit, sizeof(unlit)),
            join(needs, "; "sv));
}

// The plan for config's programs, or false with all it needs that they don't express.
bool make_plan(const gx::ShaderConfig& config, Plan& plan, Needs& needs) {
  if (config.lineMode != 0) {
    needs.push_back("lines and points"s);
  }
  if (config.tevStageCount == 0 || config.tevStageCount > gx::MaxTevStages) {
    needs.push_back(fmt::format("{} TEV stages", config.tevStageCount));
    return false; // the rest reads the stages
  }
  if (config.attrs[GX_VA_POS].attrType == GX_NONE) {
    needs.push_back("no positions"s);
  }
  for (u32 attr = GX_VA_POS; attr <= GX_VA_TEX7; ++attr) {
    const auto& mapping = config.attrs[attr];
    if (mapping.attrType != GX_NONE &&
        mapping.compType > (attr == GX_VA_CLR0 || attr == GX_VA_CLR1 ? GX_RGBA8 : GX_F32)) {
      needs.push_back(fmt::format("vertex attribute {} of component type {}", attr, mapping.compType));
    }
  }
  plan.info = gx::build_shader_info(config);
  const auto& info = plan.info;
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
      needs.push_back(fmt::format("an emboss texgen (TEXCOORD{})", i));
    } else {
      // The types and sources vertex_body writes.
      if (tcg.type != GX_TG_MTX2x4 && tcg.type != GX_TG_MTX3x4 && tcg.type != GX_TG_SRTG) {
        needs.push_back(fmt::format("texgen type {} (TEXCOORD{})", underlying(tcg.type), i));
      }
      if (tcg.src == GX_TG_BINRM || tcg.src == GX_TG_TANGENT) {
        needs.push_back(fmt::format("a binormal or tangent texgen (TEXCOORD{})", i));
      } else if (!(tcg.src >= GX_TG_TEX0 && tcg.src <= GX_TG_TEX7) && tcg.src != GX_TG_POS && tcg.src != GX_TG_NRM &&
                 tcg.src != GX_TG_COLOR0 && tcg.src != GX_TG_COLOR1) {
        needs.push_back(fmt::format("texgen source {} (TEXCOORD{})", underlying(tcg.src), i));
      }
    }
    if (tcg.type == GX_TG_MTX3x4) {
      plan.interpolants.push_back({fmt::format("tex{}_uvw", i), "vec3"sv});
    } else {
      plan.interpolants.push_back({fmt::format("tex{}_uv", i), "vec2"sv});
    }
  }
  if (info.usesFog) {
    // Clip z and w, for the fragment's depth: xost gives a fragment program no gl_FragCoord.z.
    plan.interpolants.push_back({"fog_zw"s, "vec2"sv});
  }
  if (plan.interpolants.size() > MaxInterpolants) {
    needs.push_back(fmt::format("{} interpolants", plan.interpolants.size()));
  }
  for (u32 i = 0; i < config.tevStageCount; ++i) {
    if (uses_texture_sample(config.tevStages[i]) && config.tevStages[i].texCoordId == GX_TEXCOORD_NULL) {
      needs.push_back(fmt::format("TEV stage {} sampling a texture with no coordinates", i));
    }
  }
  return needs.empty();
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
      std::string members = all[i].members;
      // View's last member, declared only where it's read, so the programs that don't read it keep their text.
      if (i == gx::UniformView && program.find("uview.depth_range") != std::string_view::npos) {
        members += "\n    vec4 depth_range; // the render viewport's znear and zfar";
      }
      blocks += fmt::format("layout(set = 0, binding = {}, std140) uniform {} {{{}\n}} {};\n", i, all[i].type, members,
                            all[i].var);
    }
  }
  return blocks;
}

// A 3x4 matrix of rows first to first + 2 in rows, times v, a vec4.
std::string mul3x4(std::string_view rows, u32 first, std::string_view v) {
  return fmt::format("vec3(dot({0}[{1}], {3}), dot({0}[{2}], {3}), dot({0}[{4}], {3}))", rows, first, first + 1, v,
                     first + 2);
}

// The same with its first row at an index the program works out, an int.
std::string mul3x4_at(std::string_view rows, std::string_view first, std::string_view v) {
  return fmt::format("vec3(dot({0}[{1}], {2}), dot({0}[{1} + 1], {2}), dot({0}[{1} + 2], {2}))", rows, first, v);
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
  if (attr == GX_VA_PNMTXIDX) {
    return "in_pnmtxidx"s;
  }
  if (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX) {
    return fmt::format("in_tex{}mtxidx", attr - GX_VA_TEX0MTXIDX);
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

// A light's vec4s in the lights piece: position, direction, colour, angle and distance coefficients.
constexpr u32 LightVec4s = 5;

// The sampled channels' lit halves, the colour or the alpha of a channel: each one's channel and whether it's the
// alpha.
std::vector<std::pair<u32, bool>> lit_channels(const gx::ShaderConfig& config, const gx::ShaderInfo& info) {
  std::vector<std::pair<u32, bool>> lit;
  for (u32 i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (!info.sampledColorChannels.test(i)) {
      continue;
    }
    for (const bool alpha : {false, true}) {
      if (config.colorChannels[i + (alpha ? GX_ALPHA0 : 0)].lightingEnabled) {
        lit.emplace_back(i, alpha);
      }
    }
  }
  return lit;
}

// A lit channel half as gx/shader.cpp's lighting_func lights it per vertex: its ambient colour plus each light the
// config's mask names, unrolled, clamped and times its material colour, in the local cc<channel>[a]_lit, the colour's
// vec4 or the alpha.
struct LitHalf {
  const gx::ColorChannelConfig* cc;
  std::string name; // cc0, or cc0a for the alpha
  bool alpha;
  std::string sum;      // the running sum: the ambient colour, then each light's sum
  std::string material; // the material colour's vec4 or alpha
};

// Light j's term of a lit half, added to its sum. light<j>_vec, _dist2, _dist and _dir and mv_nrm are the vertex
// program's.
void light_term(LitHalf& half, u32 j, std::string& body) {
  const auto& cc = *half.cc;
  const auto& name = half.name;
  const u32 light = j * LightVec4s;
  std::vector<std::string> factors;
  if (cc.attnFn == GX_AF_SPOT) {
    // The cosine as the WGSL clamps it, NaN-safe there by a select and here by the clamp's saturate, for a point
    // light's unset direction.
    body += fmt::format(
        "\n    float {0}_cos{1} = clamp(dot(light{1}_dir, ulights.lights[{2}].xyz), 0.0, 1.0);"
        "\n    float {0}_attn{1} = max(0.0, dot(ulights.lights[{3}].xyz, vec3(1.0, {0}_cos{1}, {0}_cos{1} * "
        "{0}_cos{1}))"
        " / dot(ulights.lights[{4}].xyz, vec3(1.0, light{1}_dist, light{1}_dist2)));",
        name, j, light + 1, light + 3, light + 4);
    factors.push_back(fmt::format("{}_attn{}", name, j));
  } else if (cc.attnFn == GX_AF_SPEC) {
    body += fmt::format(
        "\n    float {0}_spec{1} = clamp(dot(mv_nrm, ulights.lights[{2}].xyz), 0.0, 1.0) * step(0.0, dot(mv_nrm,"
        " light{1}_dir));"
        "\n    vec3 {0}_terms{1} = vec3(1.0, {0}_spec{1}, {0}_spec{1} * {0}_spec{1});"
        "\n    float {0}_attn{1} = max(0.0, dot(ulights.lights[{3}].xyz, {0}_terms{1}) / max(0.0, dot({4},"
        " {0}_terms{1})));",
        name, j, light + 1, light + 3,
        cc.diffFn != GX_DF_NONE ? fmt::format("normalize(ulights.lights[{}].xyz)", light + 4)
                                : fmt::format("ulights.lights[{}].xyz", light + 4));
    factors.push_back(fmt::format("{}_attn{}", name, j));
  }
  if (cc.diffFn == GX_DF_SIGN) {
    body += fmt::format("\n    float {0}_diff{1} = dot(light{1}_dir, mv_nrm);", name, j);
    factors.push_back(fmt::format("{}_diff{}", name, j));
  } else if (cc.diffFn == GX_DF_CLAMP) {
    body += fmt::format("\n    float {0}_diff{1} = max(0.0, dot(light{1}_dir, mv_nrm));", name, j);
    factors.push_back(fmt::format("{}_diff{}", name, j));
  }
  factors.push_back(fmt::format("ulights.lights[{}]{}", light + 2, half.alpha ? ".a"sv : ""sv));
  body += fmt::format("\n    {} {}_sum{} = {} + {};", half.alpha ? "float"sv : "vec4"sv, name, j, half.sum,
                      join(factors, " * "sv));
  half.sum = fmt::format("{}_sum{}", name, j);
}

// The texgens' types and sources are ones make_plan lets through. Sets normalMatrix when the program reads the
// current normal matrix from its push constant.
void vertex_body(const gx::ShaderConfig& config, const Plan& plan, std::array<bool, gx::MaxVtxAttr>& inputs,
                 std::string& body, bool& normalMatrix) {
  const auto& info = plan.info;
  // A position matrix picked per vertex is read at its first row (decode.cpp's pnmtx_row); without one, the backend
  // pushes the current one.
  const bool perVertex = config.attrs[GX_VA_PNMTXIDX].attrType != GX_NONE;
  if (perVertex) {
    body += fmt::format("\n    int pnmtx = int({});", vtx_attr(config, GX_VA_PNMTXIDX, inputs));
  }
  body += fmt::format(
      "\n    vec4 pos4 = vec4({}, 1.0);"
      "\n    vec4 mv_pos4 = vec4({}, 1.0);"
      "\n    vec4 clip_pos = vec4(dot(uview.proj[0], mv_pos4), dot(uview.proj[1], mv_pos4),"
      " dot(uview.proj[2], mv_pos4), dot(uview.proj[3], mv_pos4));"
      "\n    gl_Position = clip_pos * vec4(1.0, -1.0, 1.0, 1.0);",
      vtx_attr(config, GX_VA_POS, inputs),
      perVertex ? mul3x4_at("upos.mtx"sv, "pnmtx"sv, "pos4"sv) : mul3x4("upush.pos_mtx"sv, 0, "pos4"sv));

  // A channel's colour or alpha from its material source, or its ambient and material sources when it's lit.
  const auto source = [&](u32 channel, GXColorSrc src, u32 slot) {
    return src == GX_SRC_VTX ? vtx_attr(config, static_cast<GXAttr>(GX_VA_CLR0 + channel), inputs)
                             : fmt::format("uview.channel_colors[{}]", channel * 4 + slot);
  };
  std::vector<LitHalf> halves;
  bool normal = false;
  u32 lights = 0;
  for (const auto& [channel, alpha] : lit_channels(config, info)) {
    const auto& cc = config.colorChannels[channel + (alpha ? GX_ALPHA0 : 0)];
    const u32 slot = alpha ? 2 : 0;
    const auto part = alpha ? ".a"sv : ""sv;
    halves.push_back({&cc, fmt::format("cc{}{}", channel, alpha ? "a"sv : ""sv), alpha,
                      fmt::format("{}{}", source(channel, cc.ambSrc, slot), part),
                      fmt::format("{}{}", source(channel, cc.matSrc, slot + 1), part)});
    normal = normal || (cc.lightMask != 0 && (cc.diffFn != GX_DF_NONE || cc.attnFn == GX_AF_SPEC));
    lights |= cc.lightMask;
  }
  // The normal in view space, normalized unless it's nearly zero, as the WGSL keeps it: picked by a factor of 1 or 0,
  // as glslang makes a branch of a ?: of calls, which xost doesn't lower in a vertex program.
  if (normal) {
    normalMatrix = !perVertex;
    body += fmt::format(
        "\n    vec4 nrm4 = vec4({}, 0.0);"
        "\n    vec3 nrm_tmp = {};"
        "\n    float nrm_len2 = dot(nrm_tmp, nrm_tmp);"
        "\n    vec3 mv_nrm = nrm_tmp * mix(1.0, inversesqrt(max(nrm_len2, 1e-10)), step(1e-10, nrm_len2));",
        vtx_attr(config, GX_VA_NRM, inputs),
        perVertex ? mul3x4_at("unrm.mtx"sv, "pnmtx"sv, "nrm4"sv) : mul3x4("upush.nrm_mtx"sv, 0, "nrm4"sv));
  }
  // The lights in turn, each one's vectors then its term in each half that sums it, so a light's vectors are dead
  // before the next light's: xost gives each value a register till it's dead.
  for (u32 j = 0; j < GX::MaxLights; ++j) {
    if ((lights & 1u << j) == 0) {
      continue;
    }
    body += fmt::format(
        "\n    vec3 light{0}_vec = ulights.lights[{1}].xyz - mv_pos4.xyz;"
        "\n    float light{0}_dist2 = dot(light{0}_vec, light{0}_vec);"
        "\n    float light{0}_dist = sqrt(light{0}_dist2);"
        "\n    vec3 light{0}_dir = light{0}_vec / light{0}_dist;",
        j, j * LightVec4s);
    for (auto& half : halves) {
      if ((half.cc->lightMask & 1u << j) != 0) {
        light_term(half, j, body);
      }
    }
  }
  for (const auto& half : halves) {
    body += fmt::format("\n    {0} {1}_lit = {2} * clamp({3}, 0.0, 1.0);", half.alpha ? "float"sv : "vec4"sv, half.name,
                        half.material, half.sum);
  }

  // Each channel's colour, then its alpha from the alpha channel: a material source, or lit.
  for (u32 i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (!info.sampledColorChannels.test(i)) {
      continue;
    }
    const auto half = [&](bool alpha) {
      const auto& cc = config.colorChannels[i + (alpha ? GX_ALPHA0 : 0)];
      if (!cc.lightingEnabled) {
        return fmt::format("{}{}", source(i, cc.matSrc, alpha ? 3 : 1), alpha ? ".a"sv : ".rgb"sv);
      }
      return alpha ? fmt::format("cc{}a_lit", i) : fmt::format("cc{}_lit.rgb", i);
    };
    const auto color = half(false);
    const auto alpha = half(true);
    body += fmt::format("\n    vec4 cc{0} = vec4({1}, {2});\n    out_cc{0} = cc{0};", i, color, alpha);
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
      UNLIKELY FATAL("unplanned texgen source {}", underlying(tcg.src));
    }
    body += fmt::format("\n    vec4 tc{} = {};", i, source);
    const auto tc = fmt::format("tc{}", i);
    std::string value;
    if (tcg.type == GX_TG_MTX2x4 || tcg.type == GX_TG_MTX3x4) {
      if (info.indexAttr.test(GX_VA_TEX0MTXIDX + i)) {
        // A matrix picked per vertex: a position matrix below the slots past MaxPnMtx, a texture matrix from them, as
        // the WGSL picks it; each index held in its array, as both are read. The attribute is the slot's first row
        // (decode.cpp), and the pick's factors are 1 and 0, so the sum is the matrix picked.
        const auto row = vtx_attr(config, static_cast<GXAttr>(GX_VA_TEX0MTXIDX + i), inputs);
        body += fmt::format(
            "\n    int tc{0}_pos_row = int(min({1}, {2}.0));"
            "\n    int tc{0}_tex_row = int(clamp({1} - {3}.0, 0.0, {4}.0));"
            "\n    float tc{0}_is_tex = step({3}.0, {1});",
            i, row, (gx::MaxPnMtx - 1) * 3, gx::MaxPnMtx * 3, (gx::MaxTexMtx - 1) * 3);
        value = fmt::format("{} * (1.0 - tc{}_is_tex) + {} * tc{}_is_tex",
                            mul3x4_at("upos.mtx"sv, fmt::format("tc{}_pos_row", i), tc), i,
                            mul3x4_at("uview.tex_mtx"sv, fmt::format("tc{}_tex_row", i), tc), i);
      } else if (tcg.mtx == GX_IDENTITY) {
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
      UNLIKELY FATAL("unplanned texgen type {}", underlying(tcg.type));
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
  if (info.usesFog) {
    body += "\n    out_fog_zw = clip_pos.zw;";
  }
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
  for (u32 attr = GX_VA_PNMTXIDX; attr <= GX_VA_TEX7MTXIDX; ++attr) {
    if (inputs[attr]) {
      add(static_cast<GXAttr>(attr), location++, AttrFormat::F32x1, 4);
    }
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

std::string vertex_source(const Plan& plan, const VertexLayout& layout, const std::string& body, bool normalMatrix) {
  std::string decls =
      normalMatrix
          ? "layout(push_constant, std140) uniform Push {\n    vec4 pos_mtx[3];\n    vec4 nrm_mtx[3];\n} upush;\n"
          : "layout(push_constant, std140) uniform Push {\n    vec4 pos_mtx[3];\n} upush;\n";
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
  Fragment(const gx::ShaderConfig& config, const gx::ShaderInfo& info) : m_config(config), m_fog(info.usesFog) {
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
      const auto& a = stage.alphaPass;
      // The operands in order, as each may add a local of its own.
      const std::string colorA = color_arg(idx, stage, c.a, 'a');
      const std::string colorB = color_arg(idx, stage, c.b, 'b');
      const std::string color =
          color_op(idx, stage, colorA, colorB, color_arg(idx, stage, c.c, 'c'), color_value(idx, stage, c.d));
      // A packed compare in the alpha op, R8, GR16 or BGR24, compares the colour operands a and b as the stage reads
      // them, before it writes its colour (gx/shader.cpp).
      const bool packed = stage.alphaOp.op >= GX_TEV_COMP_R8_GT && stage.alphaOp.op <= GX_TEV_COMP_BGR24_EQ;
      const std::string alphaA = packed ? colorA : alpha_arg(idx, stage, a.a, 'a');
      const std::string alphaB = packed ? colorB : alpha_arg(idx, stage, a.b, 'b');
      const std::string alpha =
          alpha_op(idx, stage, alphaA, alphaB, alpha_arg(idx, stage, a.c, 'c'), alpha_value(idx, stage, a.d));
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
    if (m_fog) {
      fog(color);
      color = "fog_color"s;
    }
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

  static bool is_alpha_bump(GXChannelID id) noexcept { return id == GX_ALPHA_BUMP || id == GX_ALPHA_BUMPN; }

  // The WGSL's alpha_bump_sel: the bits of one of the stage's indirect texture components its format keeps, as
  // tev_mask_expr masks them, u32(v) & mask. v is 0 to 255 and the masks keep the bits from the format's lowest, so
  // the masked value is v's whole part less its remainder by that bit. GX_ALPHA_BUMPN scales it to 1.
  std::string alpha_bump(const gx::TevStage& stage) const {
    if (stage.indTexStage >= m_config.numIndStages || stage.indTexAlphaSel == GX_ITBA_OFF) {
      return "0.0"s;
    }
    std::string_view comp;
    switch (stage.indTexAlphaSel) {
      DEFAULT_FATAL("invalid indTexAlphaSel {}", underlying(stage.indTexAlphaSel));
    case GX_ITBA_S:
      comp = "x"sv;
      break;
    case GX_ITBA_T:
      comp = "y"sv;
      break;
    case GX_ITBA_U:
      comp = "z"sv;
      break;
    }
    u32 lowest = 8;
    switch (stage.indTexFormat) {
      DEFAULT_FATAL("invalid indirect format {}", underlying(stage.indTexFormat));
    case GX_ITF_8:
    case GX_ITF_3:
      lowest = 8; // 0xF8
      break;
    case GX_ITF_5:
      lowest = 32; // 0xE0
      break;
    case GX_ITF_4:
      lowest = 16; // 0xF0
      break;
    }
    const auto value = fmt::format("floor(ind_tex{}.{})", underlying(stage.indTexStage), comp);
    const auto masked = fmt::format("(({0} - mod({0}, {1}.0)) / 255.0)", value, lowest);
    return stage.channelId == GX_ALPHA_BUMPN ? fmt::format("({} * (255.0 / 248.0))", masked) : masked;
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
      if (is_alpha_bump(stage.channelId)) {
        return fmt::format("vec3({})", alpha_bump(stage));
      }
      const auto& swap = m_config.tevSwapTable[stage.tevSwapRas];
      return fmt::format("in_cc{}.{}{}{}", gx::color_channel(stage.channelId), chan_comp(swap.red),
                         chan_comp(swap.green), chan_comp(swap.blue));
    }
    case GX_CC_RASA: {
      if (stage.channelId == GX_COLOR_ZERO || stage.channelId == GX_COLOR_NULL) {
        return "vec3(0.0)"s;
      }
      if (is_alpha_bump(stage.channelId)) {
        return fmt::format("vec3({})", alpha_bump(stage));
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
      if (is_alpha_bump(stage.channelId)) {
        return alpha_bump(stage);
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

  // The 8-, 16- and 24-bit keys GX compares, rounded as the WGSL rounds them: from a colour operand's red, green and
  // red, or blue, green and red, or for RGB8 each channel of a colour operand or an alpha operand alone.
  static std::string compare_key(GXTevOp op, std::string_view v) {
    switch (op) {
    case GX_TEV_COMP_R8_GT:
    case GX_TEV_COMP_R8_EQ:
      return fmt::format("floor({}.r * 255.0 + 0.5)", v);
    case GX_TEV_COMP_GR16_GT:
    case GX_TEV_COMP_GR16_EQ:
      return fmt::format("floor(dot({}.rg * 255.0, vec2(1.0, 256.0)) + 0.5)", v);
    case GX_TEV_COMP_BGR24_GT:
    case GX_TEV_COMP_BGR24_EQ:
      return fmt::format("floor(dot({}.rgb * 255.0, vec3(1.0, 256.0, 65536.0)) + 0.5)", v);
    default:
      return fmt::format("floor({} * 255.0 + 0.5)", v);
    }
  }

  static bool is_greater(GXTevOp op) {
    return op == GX_TEV_COMP_R8_GT || op == GX_TEV_COMP_GR16_GT || op == GX_TEV_COMP_BGR24_GT ||
           op == GX_TEV_COMP_RGB8_GT;
  }

  // The op before its clamp, scalar for an alpha op. A compare picks c or 0 by a factor of 1 or 0 (step), which xost
  // compiles where it doesn't compile the WGSL's select (xost: a branch writes one variable once). The keys are whole
  // numbers, so 0.5 separates them.
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
      const auto ka = compare_key(tevOp.op, a);
      const auto kb = compare_key(tevOp.op, b);
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
  // alpha its stage didn't clamp is clamped first. False for a compare that discards every fragment.
  bool alpha_compare(const std::string& alpha, bool normalized) {
    const auto& ac = m_config.alphaCompare;
    const Pass pass = combine(ac.op, compare(ac.comp0, ac.ref0), compare(ac.comp1, ac.ref1));
    if (pass.constant == 1) {
      return true;
    }
    if (pass.constant == 0) {
      return false;
    }
    m_body += fmt::format(
        "\n    // Alpha compare"
        "\n    float alpha_byte = floor({} * 255.0 + 0.5);"
        "\n    float alpha_pass = {};"
        "\n    if (alpha_pass < 0.5) {{\n        discard;\n    }}",
        normalized ? alpha : fmt::format("clamp({}, 0.0, 1.0)", alpha), pass.expr);
    return true;
  }

  // The WGSL's fog, mixed into the stages' colour as fog_color. The fragment's depth comes from clip z and w, as the
  // GPU's viewport maps it, and range fog's factor from the table gx/command_processor.cpp's build_fog_range_lut makes
  // for each column, as the WGSL reads it: one sample of set 2's copy of it (glsl.hpp). gl_FragCoord.x is the column
  // plus 0.5, so x / width, wrapped by the sampler, is the middle of the column's texel along u, and the middle of its
  // row is worked out from the whole part.
  void fog(std::string_view color) {
    m_body += gx::UseReversedZ ? "\n    // Fog"
                                 "\n    float fog_ndc = in_fog_zw.x / in_fog_zw.y;"
                                 "\n    float fog_depth = uview.depth_range.y - fog_ndc * (uview.depth_range.y -"
                                 " uview.depth_range.x);"sv
                               : "\n    // Fog"
                                 "\n    float fog_ndc = in_fog_zw.x / in_fog_zw.y;"
                                 "\n    float fog_depth = uview.depth_range.x + fog_ndc * (uview.depth_range.y -"
                                 " uview.depth_range.x);"sv;
    m_body += (m_config.fogType & 0x08) != 0
                  ? "\n    float fog_base = uview.fog[1].x * fog_depth;"sv
                  : "\n    float fog_base = uview.fog[1].x / (uview.fog[1].y - fog_depth);"sv;
    if (m_config.fogRangeEnabled) {
      m_body += fmt::format(
          "\n    float fog_column = gl_FragCoord.x * (1.0 / {0}.0);"
          "\n    vec4 fog_range = texture(fog_range_table, vec2(fog_column, floor(fog_column) * (1.0 / {1}.0) +"
          " (0.5 / {1}.0)));"
          "\n    float fog_f = clamp(fog_base * fog_range.x - uview.fog[1].z, 0.0, 1.0);",
          FogRangeTableWidth, FogRangeTableHeight);
    } else {
      m_body += "\n    float fog_f = clamp(fog_base - uview.fog[1].z, 0.0, 1.0);"sv;
    }
    switch (m_config.fogType & 0x07) {
      DEFAULT_FATAL("invalid fog type {}", m_config.fogType);
    case GX_FOG_PERSP_LIN:
      m_body += "\n    float fog_z = fog_f;"sv;
      break;
    case GX_FOG_PERSP_EXP:
      m_body += "\n    float fog_z = 1.0 - exp2(-8.0 * fog_f);"sv;
      break;
    case GX_FOG_PERSP_EXP2:
      m_body += "\n    float fog_z = 1.0 - exp2(-8.0 * fog_f * fog_f);"sv;
      break;
    case GX_FOG_PERSP_REVEXP:
      m_body += "\n    float fog_z = exp2(-8.0 * (1.0 - fog_f));"sv;
      break;
    case GX_FOG_PERSP_REVEXP2:
      m_body += "\n    float fog_g = 1.0 - fog_f;\n    float fog_z = exp2(-8.0 * fog_g * fog_g);"sv;
      break;
    }
    m_body += fmt::format("\n    vec3 fog_color = mix({}, uview.fog[0].rgb, clamp(fog_z, 0.0, 1.0));", color);
  }

  const gx::ShaderConfig& m_config;
  bool m_fog;
  std::array<Reg, gx::MaxTevRegs> m_regs;
  std::string m_body;
};

std::string_view ind_scale(GXIndTexScale scale) {
  switch (scale) {
    DEFAULT_FATAL("unhandled indirect scale {}", underlying(scale));
  case GX_ITS_1:
    return "1.0"sv;
  case GX_ITS_2:
    return "(1.0 / 2.0)"sv;
  case GX_ITS_4:
    return "(1.0 / 4.0)"sv;
  case GX_ITS_8:
    return "(1.0 / 8.0)"sv;
  case GX_ITS_16:
    return "(1.0 / 16.0)"sv;
  case GX_ITS_32:
    return "(1.0 / 32.0)"sv;
  case GX_ITS_64:
    return "(1.0 / 64.0)"sv;
  case GX_ITS_128:
    return "(1.0 / 128.0)"sv;
  case GX_ITS_256:
    return "(1.0 / 256.0)"sv;
  }
}

// Each indirect stage the TEV stages use, its texture sampled once, as the WGSL samples it: at its coordinates scaled
// first by the texture coordinate scale, then by the stage's, back to the texture's own; its components 0 to 255, in
// the order alpha, blue, green.
void indirect_textures(const gx::ShaderConfig& config, const gx::ShaderInfo& info, std::string& pre) {
  for (u32 n = 0; n < info.usedIndStages.size(); ++n) {
    if (!info.usedIndStages.test(n)) {
      continue;
    }
    const auto& indStage = config.indStages[n];
    const u32 coord = underlying(indStage.texCoordId);
    const u32 map = underlying(indStage.texMapId);
    pre += fmt::format(
        "\n    // Indirect stage {0}"
        "\n    vec3 ind_tex{0} = 255.0 * texture(tex{1}, tex{2}_uv * umat.texcoord_scale[{2}].xy * vec2({3}, {4}) /"
        " umat.tex_size_bias[{1}].xy).abg;",
        n, map, coord, ind_scale(indStage.scaleS), ind_scale(indStage.scaleT));
  }
}

// The WGSL's % on f32, which truncates: a wrap of a coordinate, in texels.
std::string ind_wrap(GXIndTexWrap wrap, const std::string& coord) {
  switch (wrap) {
    DEFAULT_FATAL("unhandled indirect wrap {}", underlying(wrap));
  case GX_ITW_OFF:
    return coord;
  case GX_ITW_256:
  case GX_ITW_128:
  case GX_ITW_64:
  case GX_ITW_32:
  case GX_ITW_16: {
    static constexpr std::array Sizes{0, 256, 128, 64, 32, 16};
    const int size = Sizes[wrap];
    return fmt::format("({0} - {1}.0 * trunc({0} / {1}.0))", coord, size);
  }
  case GX_ITW_0:
    return "0.0"s;
  }
}

// TEV stage idx's texture coordinate from indirect texturing, as the WGSL works it out: the stage's indirect
// coordinate through its format, bias and matrix, added to its own coordinate, wrapped, and to the running coordinate
// texCoord before it (the WGSL's t_TexCoord) where the stage adds to it. The uv the stage samples its texture at, or
// empty where it samples at its own coordinate.
std::string indirect_coord(const gx::ShaderConfig& config, const gx::ShaderInfo& info, u32 idx, std::string& texCoord,
                           std::string& pre) {
  const auto& stage = config.tevStages[idx];
  const bool needsIndirectCoord = stage.indTexMtxId != GX_ITM_OFF;
  const bool hasIndirectStage = stage.indTexStage < config.numIndStages;
  const bool needsTevTexCoord =
      needsIndirectCoord || stage.indTexWrapS != GX_ITW_OFF || stage.indTexWrapT != GX_ITW_OFF || stage.indTexAddPrev;
  if (!needsTevTexCoord) {
    return {};
  }
  const bool hasBaseTexture = stage.texMapId != GX_TEXMAP_NULL;
  const bool hasBaseCoord = stage.texCoordId != GX_TEXCOORD_NULL && hasBaseTexture;
  const u32 coord = underlying(stage.texCoordId);
  pre += fmt::format("\n    // TEV stage {} indirect", idx);

  // The indirect matrix's offset, in texels.
  std::string offset;
  if (needsIndirectCoord && hasIndirectStage) {
    std::string_view shift;
    switch (stage.indTexFormat) {
      DEFAULT_FATAL("unhandled indirect format {}", underlying(stage.indTexFormat));
    case GX_ITF_8:
      break;
    case GX_ITF_5:
      shift = " / 8.0"sv;
      break;
    case GX_ITF_4:
      shift = " / 16.0"sv;
      break;
    case GX_ITF_3:
      shift = " / 32.0"sv;
      break;
    }
    const u32 indStage = underlying(stage.indTexStage);
    if (shift.empty()) {
      pre += fmt::format("\n    vec3 ind{0}_coord = ind_tex{1};", idx, indStage);
    } else {
      pre += fmt::format("\n    vec3 ind{0}_coord = floor(ind_tex{1}{2});", idx, indStage, shift);
    }
    std::string indCoord = fmt::format("ind{}_coord", idx);
    if (stage.indTexBiasSel != GX_ITB_NONE) {
      const auto sel = stage.indTexBiasSel;
      const bool s = sel == GX_ITB_S || sel == GX_ITB_ST || sel == GX_ITB_SU || sel == GX_ITB_STU;
      const bool t = sel == GX_ITB_T || sel == GX_ITB_ST || sel == GX_ITB_TU || sel == GX_ITB_STU;
      const bool u = sel == GX_ITB_U || sel == GX_ITB_SU || sel == GX_ITB_TU || sel == GX_ITB_STU;
      pre += fmt::format("\n    vec3 ind{0}_biased = ind{0}_coord + vec3({1}, {2}, {3}) * {4};", idx,
                         s ? "1.0"sv : "0.0"sv, t ? "1.0"sv : "0.0"sv, u ? "1.0"sv : "0.0"sv,
                         stage.indTexFormat == GX_ITF_8 ? "-128.0"sv : "1.0"sv);
      indCoord = fmt::format("ind{}_biased", idx);
    }
    if (stage.indTexMtxId >= GX_ITM_0 && stage.indTexMtxId <= GX_ITM_2) {
      // Static 2x3 matrix: dot(mat_row, vec3(S, T, U)) times its scale.
      const u32 mtx = stage.indTexMtxId - GX_ITM_0;
      pre += fmt::format("\n    vec4 ind{0}_c0 = uview.ind_mtx[{1}];\n    vec4 ind{0}_c1 = uview.ind_mtx[{2}];", idx,
                         mtx * 2, mtx * 2 + 1);
      offset = fmt::format(
          "vec2(dot(vec3(ind{0}_c0.xz, ind{0}_c1.x), {1}), dot(vec3(ind{0}_c0.yw, ind{0}_c1.y), {1})) * ind{0}_c1.z",
          idx, indCoord);
    } else if (stage.indTexMtxId >= GX_ITM_S0 && stage.indTexMtxId <= GX_ITM_S2 && hasBaseCoord) {
      // Dynamic S: the scaled texture coordinate times the indirect S, by the matrix's scale over 256.
      offset = fmt::format("tex{0}_uv * umat.texcoord_scale[{0}].xy * {1}.x * uview.ind_mtx[{2}].z / 256.0", coord,
                           indCoord, (stage.indTexMtxId - GX_ITM_S0) * 2 + 1);
    } else if (stage.indTexMtxId >= GX_ITM_T0 && stage.indTexMtxId <= GX_ITM_T2 && hasBaseCoord) {
      offset = fmt::format("tex{0}_uv * umat.texcoord_scale[{0}].xy * {1}.y * uview.ind_mtx[{2}].z / 256.0", coord,
                           indCoord, (stage.indTexMtxId - GX_ITM_T0) * 2 + 1);
    }
  }

  // The stage's own coordinate, in texels unless nothing moves it, wrapped, then the offset added.
  const bool useSimpleCoords = stage.indTexMtxId == GX_ITM_OFF && !stage.indTexAddPrev;
  std::string base;
  if (hasBaseCoord) {
    if (useSimpleCoords) {
      base = fmt::format("tex{}_uv", coord);
    } else {
      pre += fmt::format("\n    vec2 ind{0}_texel = tex{1}_uv * umat.texcoord_scale[{1}].xy;", idx, coord);
      base = fmt::format("ind{}_texel", idx);
    }
  }
  std::string wrapped = base;
  if (!base.empty() && (stage.indTexWrapS != GX_ITW_OFF || stage.indTexWrapT != GX_ITW_OFF)) {
    wrapped =
        fmt::format("vec2({}, {})", ind_wrap(stage.indTexWrapS, base + ".x"), ind_wrap(stage.indTexWrapT, base + ".y"));
  }
  std::string coordExpr;
  if (!wrapped.empty() && !offset.empty()) {
    coordExpr = fmt::format("{} + ({})", wrapped, offset);
  } else if (!wrapped.empty()) {
    coordExpr = wrapped;
  } else {
    coordExpr = offset;
  }
  if (!info.usedIndStages.any() || coordExpr.empty()) {
    return {};
  }
  if (stage.indTexAddPrev) {
    pre += fmt::format("\n    vec2 tev_coord{} = {} + ({});", idx, texCoord, coordExpr);
  } else {
    pre += fmt::format("\n    vec2 tev_coord{} = {};", idx, coordExpr);
  }
  texCoord = fmt::format("tev_coord{}", idx);
  if (!uses_texture_sample(stage) || !hasBaseTexture) {
    return {};
  }
  if (useSimpleCoords) {
    pre += fmt::format("\n    vec2 ind{}_uv = {};", idx, texCoord);
  } else {
    pre +=
        fmt::format("\n    vec2 ind{}_uv = {} / umat.tex_size_bias[{}].xy;", idx, texCoord, underlying(stage.texMapId));
  }
  return fmt::format("ind{}_uv", idx);
}

// Whether the fragment program has range fog, which reads the range fog table.
bool reads_fog_range(const gx::ShaderConfig& config, const gx::ShaderInfo& info) noexcept {
  return info.usesFog && config.fogRangeEnabled;
}

// False when the alpha compare discards every fragment.
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
  indirect_textures(config, info, pre);
  std::string texCoord = "vec2(0.0)"s;
  for (u32 i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    const auto uv = indirect_coord(config, info, i, texCoord, pre);
    if (!uses_texture_sample(stage)) {
      continue;
    }
    // xost takes only a bias known when it compiles, so the material's LOD bias goes in the sampler.
    pre += fmt::format("\n    vec4 sampled{0} = texture(tex{1}, {2});", i, underlying(stage.texMapId),
                       uv.empty() ? fmt::format("tex{}_uv", underlying(stage.texCoordId)) : uv);
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
  if (reads_fog_range(config, info)) {
    decls += fmt::format("layout(set = {}, binding = 0) uniform sampler2D fog_range_table;\n", FogRangeSet);
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
  Needs needs;
  if (!make_plan(config, plan, needs)) {
    refuse(config, needs);
    return {};
  }
  Programs programs;
  std::array<bool, gx::MaxVtxAttr> inputs{};
  std::string body;
  vertex_body(config, plan, inputs, body, programs.normalMatrix);
  if (!fragment_source(config, plan, programs.fragment)) {
    // Nothing is drawn; xost compiles no program that only discards, and the backend skips such a draw.
    refuse(config, {"an alpha compare that discards every fragment"s});
    return {};
  }
  programs.layout = make_layout(inputs);
  programs.vertex = vertex_source(plan, programs.layout, body, programs.normalMatrix);
  programs.fogRange = reads_fog_range(config, plan.info);
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
