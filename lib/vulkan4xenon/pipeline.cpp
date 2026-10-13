// The vulkan4xenon backend's programs and pipelines: GLSL compiled offline and found in the game's pack, the pipeline
// layouts, and the pipelines for clears, EFB copies and the present.
#include "backend.hpp"

#include "../gx/gx.hpp"

#include <aurora/glsl_pack.h>

#include <absl/container/flat_hash_map.h>
#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

// Without a pack linked in, none: every program is reported missing.
extern "C" __attribute__((weak)) const AuroraGlslPack aurora_glsl_pack{};

namespace aurora::gfx::vulkan4xenon {
namespace {
using Program = std::span<const uint32_t>;

// The full-screen triangle's: positions in clip space, and texture coordinates from the top left.
constexpr std::array<float, 12> TriangleVertices{
    -1.f, -1.f, 0.f, 0.f, //
    3.f,  -1.f, 2.f, 0.f, //
    -1.f, 3.f,  0.f, 2.f, //
};
constexpr uint32_t TriangleStride = 4 * sizeof(float);

// Pipelines are made one at a time, which also guards the missing programs logged.
std::mutex g_pipelineMutex;
std::unordered_set<uint64_t> g_missing;

VkDescriptorSetLayout g_uniformSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout g_textureSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout g_sourceSetLayout = VK_NULL_HANDLE;
VkPipelineLayout g_gxLayout = VK_NULL_HANDLE;
VkPipelineLayout g_clearLayout = VK_NULL_HANDLE;
VkPipelineLayout g_copyLayout = VK_NULL_HANDLE;
Buffer g_triangle;
VkPipeline g_presentPipeline = VK_NULL_HANDLE;

std::string g_clearVertex;
std::string g_clearFragment;
std::string g_copyVertex;
// Render worker only. By format, and whether the copy reads alpha as 1.
absl::flat_hash_map<std::pair<GXTexFmt, bool>, VkPipeline> g_copyPipelines;

// The SPIR-V the offline step compiled for a program, from the game's pack: none, logged once, for a program it
// doesn't hold, whose pipeline isn't made and whose draws are skipped. Under g_pipelineMutex.
Program load_program(const std::string& source, bool fragment) noexcept {
  const uint64_t hash = glsl::program_hash(source, fragment);
  const std::span programs{aurora_glsl_pack.programs, aurora_glsl_pack.count};
  const auto found = std::ranges::lower_bound(programs, hash, {}, &AuroraGlslProgram::hash);
  if (found != programs.end() && found->hash == hash) {
    return {found->spirv, found->bytes / sizeof(uint32_t)};
  }
  if (g_missing.insert(hash).second) {
    Log.error("{:016x}.{} isn't in the shader pack: record it on the null backend and rebuild the pack", hash,
              fragment ? "frag" : "vert");
  }
  return {};
}

bool create_module(Program program, VkShaderModule& out) noexcept {
  const VkShaderModuleCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = program.size_bytes(),
      .pCode = program.data(),
  };
  return succeeded(vkCreateShaderModule(g_device, &info, nullptr, &out), "vkCreateShaderModule");
}

struct PipelineDesc {
  Program vertex;
  Program fragment;
  VkPipelineLayout layout;
  VkRenderPass renderPass;
  uint32_t stride;
  std::span<const VkVertexInputAttributeDescription> attributes;
  VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
  bool depthTest = false;
  bool depthWrite = false;
  VkCompareOp depthCompare = VK_COMPARE_OP_ALWAYS;
  VkPipelineColorBlendAttachmentState blend{
      .colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  };
};

// Under g_pipelineMutex.
VkPipeline create_pipeline(const PipelineDesc& desc) noexcept {
  std::array<VkShaderModule, 2> modules{};
  if (!create_module(desc.vertex, modules[0]) || !create_module(desc.fragment, modules[1])) {
    for (const auto module : modules) {
      vkDestroyShaderModule(g_device, module, nullptr);
    }
    return VK_NULL_HANDLE;
  }
  const std::array stages{
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = modules[0],
          .pName = "main",
      },
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = modules[1],
          .pName = "main",
      },
  };
  const VkVertexInputBindingDescription binding{
      .binding = 0,
      .stride = desc.stride,
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
  };
  const VkPipelineVertexInputStateCreateInfo vertexInput{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &binding,
      .vertexAttributeDescriptionCount = static_cast<uint32_t>(desc.attributes.size()),
      .pVertexAttributeDescriptions = desc.attributes.data(),
  };
  const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };
  const VkPipelineViewportStateCreateInfo viewport{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
  };
  // Aurora's front face (webgpu/backend.cpp, to_primitive_state).
  const VkPipelineRasterizationStateCreateInfo rasterization{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = desc.cullMode,
      .frontFace = VK_FRONT_FACE_CLOCKWISE,
      .lineWidth = 1.f,
  };
  const VkPipelineMultisampleStateCreateInfo multisample{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };
  const VkPipelineDepthStencilStateCreateInfo depthStencil{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = desc.depthTest,
      .depthWriteEnable = desc.depthWrite,
      .depthCompareOp = desc.depthCompare,
  };
  const VkPipelineColorBlendStateCreateInfo blend{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &desc.blend,
  };
  const std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  const VkPipelineDynamicStateCreateInfo dynamic{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = dynamicStates.size(),
      .pDynamicStates = dynamicStates.data(),
  };
  const VkGraphicsPipelineCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = stages.size(),
      .pStages = stages.data(),
      .pVertexInputState = &vertexInput,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewport,
      .pRasterizationState = &rasterization,
      .pMultisampleState = &multisample,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &blend,
      .pDynamicState = &dynamic,
      .layout = desc.layout,
      .renderPass = desc.renderPass,
  };
  VkPipeline pipeline = VK_NULL_HANDLE;
  succeeded(vkCreateGraphicsPipelines(g_device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline),
            "vkCreateGraphicsPipelines");
  // The pipeline holds what it needs of the modules, which can go.
  for (const auto module : modules) {
    vkDestroyShaderModule(g_device, module, nullptr);
  }
  return pipeline;
}

bool create_set_layout(std::span<const VkDescriptorSetLayoutBinding> bindings, VkDescriptorSetLayout& out) noexcept {
  const VkDescriptorSetLayoutCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = static_cast<uint32_t>(bindings.size()),
      .pBindings = bindings.data(),
  };
  return succeeded(vkCreateDescriptorSetLayout(g_device, &info, nullptr, &out), "vkCreateDescriptorSetLayout");
}

bool create_layout(std::span<const VkDescriptorSetLayout> sets, VkShaderStageFlags pushStage, uint32_t pushSize,
                   VkPipelineLayout& out) noexcept {
  const VkPushConstantRange push{.stageFlags = pushStage, .offset = 0, .size = pushSize};
  const VkPipelineLayoutCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = static_cast<uint32_t>(sets.size()),
      .pSetLayouts = sets.data(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push,
  };
  return succeeded(vkCreatePipelineLayout(g_device, &info, nullptr, &out), "vkCreatePipelineLayout");
}

bool create_layouts() noexcept {
  // Binding i is uniform piece i (gx::UniformPiece), as glsl.hpp's programs declare them.
  std::array<VkDescriptorSetLayoutBinding, gx::UniformPieceCount> uniformBindings{};
  for (uint32_t i = 0; i < uniformBindings.size(); ++i) {
    uniformBindings[i] = {
        .binding = i,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };
  }
  std::array<VkDescriptorSetLayoutBinding, gx::MaxTextures> textureBindings{};
  for (uint32_t i = 0; i < textureBindings.size(); ++i) {
    textureBindings[i] = {
        .binding = i,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
  }
  if (!create_set_layout(uniformBindings, g_uniformSetLayout) ||
      !create_set_layout(textureBindings, g_textureSetLayout) ||
      !create_set_layout(std::span{textureBindings}.first(1), g_sourceSetLayout)) {
    return false;
  }
  const std::array gxSets{g_uniformSetLayout, g_textureSetLayout};
  // The GX programs' current position and normal matrices, the clear's colour, and a copy's source rectangle.
  return create_layout(gxSets, VK_SHADER_STAGE_VERTEX_BIT, 2 * sizeof(Mat3x4<float>), g_gxLayout) &&
         create_layout({}, VK_SHADER_STAGE_FRAGMENT_BIT, 16, g_clearLayout) &&
         create_layout(std::span{&g_sourceSetLayout, 1}, VK_SHADER_STAGE_VERTEX_BIT, 16, g_copyLayout);
}

// GX's blend, depth and cull state as Dawn's backend sets it (webgpu/backend.cpp, build_pipeline).
VkBlendFactor blend_factor(GXBlendFactor factor, bool destination) noexcept {
  switch (factor) {
  case GX_BL_ZERO:
    return VK_BLEND_FACTOR_ZERO;
  case GX_BL_ONE:
    return VK_BLEND_FACTOR_ONE;
  case GX_BL_SRCCLR: // GX_BL_DSTCLR as a source factor
    return destination ? VK_BLEND_FACTOR_SRC_COLOR : VK_BLEND_FACTOR_DST_COLOR;
  case GX_BL_INVSRCCLR: // GX_BL_INVDSTCLR as a source factor
    return destination ? VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR : VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
  case GX_BL_SRCALPHA:
    return VK_BLEND_FACTOR_SRC_ALPHA;
  case GX_BL_INVSRCALPHA:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  case GX_BL_DSTALPHA:
    return VK_BLEND_FACTOR_DST_ALPHA;
  case GX_BL_INVDSTALPHA:
    return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
  }
  return VK_BLEND_FACTOR_ONE;
}

VkPipelineColorBlendAttachmentState blend_state(const gx::PipelineConfig& config) noexcept {
  VkPipelineColorBlendAttachmentState state{.colorBlendOp = VK_BLEND_OP_ADD};
  switch (config.blendMode) {
  case GX_BM_BLEND:
    state.blendEnable = VK_TRUE;
    state.srcColorBlendFactor = blend_factor(config.blendFacSrc, false);
    state.dstColorBlendFactor = blend_factor(config.blendFacDst, true);
    break;
  case GX_BM_SUBTRACT:
    state.blendEnable = VK_TRUE;
    state.colorBlendOp = VK_BLEND_OP_REVERSE_SUBTRACT;
    state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    break;
  case GX_BM_LOGIC:
    // Dawn's three logic ops, as blending; unsupported refuses the others.
    state.blendEnable = VK_TRUE;
    state.srcColorBlendFactor = config.blendOp == GX_LO_COPY ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_ZERO;
    state.dstColorBlendFactor = config.blendOp == GX_LO_NOOP ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_ZERO;
    break;
  default:
    break;
  }
  state.alphaBlendOp = state.colorBlendOp;
  state.srcAlphaBlendFactor = state.srcColorBlendFactor;
  state.dstAlphaBlendFactor = state.dstColorBlendFactor;
  if (config.colorUpdate) {
    state.colorWriteMask |= VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
  }
  if (config.alphaUpdate) {
    state.colorWriteMask |= VK_COLOR_COMPONENT_A_BIT;
  }
  return state;
}

VkCompareOp compare_op(GXCompare compare) noexcept {
  switch (compare) {
  case GX_NEVER:
    return VK_COMPARE_OP_NEVER;
  case GX_LESS:
    return gx::UseReversedZ ? VK_COMPARE_OP_GREATER : VK_COMPARE_OP_LESS;
  case GX_EQUAL:
    return VK_COMPARE_OP_EQUAL;
  case GX_LEQUAL:
    return gx::UseReversedZ ? VK_COMPARE_OP_GREATER_OR_EQUAL : VK_COMPARE_OP_LESS_OR_EQUAL;
  case GX_GREATER:
    return gx::UseReversedZ ? VK_COMPARE_OP_LESS : VK_COMPARE_OP_GREATER;
  case GX_NEQUAL:
    return VK_COMPARE_OP_NOT_EQUAL;
  case GX_GEQUAL:
    return gx::UseReversedZ ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_GREATER_OR_EQUAL;
  case GX_ALWAYS:
    break;
  }
  return VK_COMPARE_OP_ALWAYS;
}

// GX_CULL_ALL culls back faces, as Dawn's backend does: populate_pipeline_config turns every write off for it, so what
// it doesn't cull draws nothing.
VkCullModeFlags cull_mode(GXCullMode mode) noexcept {
  switch (mode) {
  case GX_CULL_FRONT:
    return VK_CULL_MODE_FRONT_BIT;
  case GX_CULL_BACK:
  case GX_CULL_ALL:
    return VK_CULL_MODE_BACK_BIT;
  default:
    return VK_CULL_MODE_NONE;
  }
}

// What the layer can't draw as GX does, or nullptr. Such a config is refused, and its draws are skipped, rather than
// drawn without it. The layer has no constant blend colour, dual-source blending or logic ops (vulkan4xenon's
// docs/SUPPORT.md, Pipelines), so destination alpha, which Dawn's backend draws with a blend constant and, when the
// colour blend reads source alpha, dual-source blending or an alpha prepass, isn't drawn. Its depth bias is unverified
// against GX's polygon offset.
const char* unsupported(const gx::PipelineConfig& config) noexcept {
  if (config.dstAlpha != UINT32_MAX) {
    return "destination alpha (GXSetDstAlpha)";
  }
  if (config.blendMode == GX_BM_LOGIC && config.blendOp != GX_LO_CLEAR && config.blendOp != GX_LO_COPY &&
      config.blendOp != GX_LO_NOOP) {
    return "a logic op other than clear, copy and noop";
  }
  if (config.polygonOffsetBits != 0 || config.polygonOffsetScaleBits != 0) {
    return "polygon offset";
  }
  return nullptr;
}

VkFormat attribute_format(decode::AttrFormat format) noexcept {
  switch (format) {
  case decode::AttrFormat::F32x1:
    return VK_FORMAT_R32_SFLOAT;
  case decode::AttrFormat::F32x2:
    return VK_FORMAT_R32G32_SFLOAT;
  case decode::AttrFormat::F32x3:
    return VK_FORMAT_R32G32B32_SFLOAT;
  case decode::AttrFormat::U8x4:
    break;
  }
  return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
}

std::array<VkVertexInputAttributeDescription, 2> triangle_attributes() noexcept {
  return {
      VkVertexInputAttributeDescription{.location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = 0},
      VkVertexInputAttributeDescription{
          .location = 1, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = 2 * sizeof(float)},
  };
}

// A copy's pipeline into a render pass of the target's format; null when its programs aren't in the pack.
VkPipeline make_copy_pipeline(GXTexFmt format, bool opaqueAlpha, VkRenderPass renderPass) noexcept {
  const auto attributes = triangle_attributes();
  std::lock_guard lock{g_pipelineMutex};
  const auto vertex = load_program(g_copyVertex, false);
  const auto fragment = load_program(glsl::copy_fragment(format, opaqueAlpha), true);
  if (vertex.empty() || fragment.empty()) {
    return VK_NULL_HANDLE;
  }
  return create_pipeline({
      .vertex = vertex,
      .fragment = fragment,
      .layout = g_copyLayout,
      .renderPass = renderPass,
      .stride = TriangleStride,
      .attributes = attributes,
  });
}
} // namespace

bool initialize_pipelines() noexcept {
  if (!create_layouts() ||
      !create_buffer(sizeof(TriangleVertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false, "triangle", g_triangle)) {
    shutdown_pipelines();
    return false;
  }
  std::memcpy(g_triangle.mapped, TriangleVertices.data(), sizeof(TriangleVertices));
  g_clearVertex = glsl::clear_vertex();
  g_clearFragment = glsl::clear_fragment();
  g_copyVertex = glsl::copy_vertex();
  Log.info("Shader pack: {} programs", aurora_glsl_pack.count);
  // The present copies the EFB to the swapchain's B8G8R8A8 images.
  g_presentPipeline = make_copy_pipeline(GX_TF_RGBA8, false, present_pass(Load::DontCare));
  return true;
}

void shutdown_pipelines() noexcept {
  for (const auto& [format, pipeline] : g_copyPipelines) {
    vkDestroyPipeline(g_device, pipeline, nullptr);
  }
  g_copyPipelines.clear();
  vkDestroyPipeline(g_device, g_presentPipeline, nullptr);
  g_presentPipeline = VK_NULL_HANDLE;
  for (auto* layout : {&g_gxLayout, &g_clearLayout, &g_copyLayout}) {
    vkDestroyPipelineLayout(g_device, *layout, nullptr);
    *layout = VK_NULL_HANDLE;
  }
  for (auto* layout : {&g_uniformSetLayout, &g_textureSetLayout, &g_sourceSetLayout}) {
    vkDestroyDescriptorSetLayout(g_device, *layout, nullptr);
    *layout = VK_NULL_HANDLE;
  }
  destroy_buffer(g_triangle);
}

VkDescriptorSetLayout uniform_set_layout() noexcept { return g_uniformSetLayout; }

VkDescriptorSetLayout texture_set_layout() noexcept { return g_textureSetLayout; }

VkDescriptorSetLayout source_set_layout() noexcept { return g_sourceSetLayout; }

VkPipelineLayout copy_layout() noexcept { return g_copyLayout; }

VkBuffer triangle_buffer() noexcept { return g_triangle.buffer; }

std::shared_ptr<backend::Pipeline> make_pipeline(const clear::PipelineConfig& config,
                                                 const RenderTargetLayout& layout) noexcept {
  auto pipeline = std::make_shared<backend::Pipeline>();
  pipeline->layout = g_clearLayout;
  const bool depth = layout.depthStencilFormat != TextureFormat::Undefined;
  VkColorComponentFlags writeMask = 0;
  if (config.clearColor) {
    writeMask |= VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
  }
  if (config.clearAlpha) {
    writeMask |= VK_COLOR_COMPONENT_A_BIT;
  }
  const auto attributes = triangle_attributes();
  std::lock_guard lock{g_pipelineMutex};
  const auto vertex = load_program(g_clearVertex, false);
  const auto fragment = load_program(g_clearFragment, true);
  if (vertex.empty() || fragment.empty()) {
    return nullptr;
  }
  pipeline->pipeline = create_pipeline({
      .vertex = vertex,
      .fragment = fragment,
      .layout = g_clearLayout,
      .renderPass = depth ? scene_pass(Load::Load, Load::Load) : color_pass(Load::Load),
      .stride = TriangleStride,
      .attributes = std::span{attributes}.first(1),
      .depthTest = depth,
      .depthWrite = depth && config.clearDepth,
      .blend = {.colorWriteMask = writeMask},
  });
  return pipeline->pipeline != VK_NULL_HANDLE ? pipeline : nullptr;
}

std::shared_ptr<backend::Pipeline> make_pipeline(const gx::PipelineConfig& config) noexcept {
  if (const char* reason = unsupported(config)) {
    Log.error("vulkan4xenon can't draw {}; a pipeline's draws are skipped", reason);
    return nullptr;
  }
  auto pipeline = std::make_shared<backend::Pipeline>();
  pipeline->layout = g_gxLayout;
  pipeline->shaderConfig = config.shaderConfig;
  auto programs = glsl::build_programs(config.shaderConfig);
  if (programs.vertex.empty() || programs.fragment.empty()) {
    return nullptr;
  }
  pipeline->vertexLayout = std::move(programs.layout);
  pipeline->decodeFormat = decode::decode_format(pipeline->shaderConfig, pipeline->vertexLayout);
  pipeline->normalMatrix = programs.normalMatrix;
  std::vector<VkVertexInputAttributeDescription> attributes;
  for (const auto& attribute : pipeline->vertexLayout.attributes) {
    attributes.push_back({
        .location = attribute.location,
        .binding = 0,
        .format = attribute_format(attribute.format),
        .offset = attribute.offset,
    });
  }
  std::lock_guard lock{g_pipelineMutex};
  const auto vertex = load_program(programs.vertex, false);
  const auto fragment = load_program(programs.fragment, true);
  if (vertex.empty() || fragment.empty()) {
    return nullptr;
  }
  pipeline->pipeline = create_pipeline({
      .vertex = vertex,
      .fragment = fragment,
      .layout = g_gxLayout,
      .renderPass = scene_pass(Load::Load, Load::Load),
      .stride = pipeline->vertexLayout.stride,
      .attributes = attributes,
      .cullMode = cull_mode(config.cullMode),
      .depthTest = config.depthCompare,
      .depthWrite = config.depthCompare && config.depthUpdate,
      .depthCompare = compare_op(config.depthFunc),
      .blend = blend_state(config),
  });
  return pipeline->pipeline != VK_NULL_HANDLE ? pipeline : nullptr;
}

bool converts(GXTexFmt format) noexcept { return glsl::converts(format); }

VkPipeline copy_pipeline(GXTexFmt format, bool opaqueAlpha) noexcept {
  if (const auto it = g_copyPipelines.find({format, opaqueAlpha}); it != g_copyPipelines.end()) {
    return it->second;
  }
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (glsl::copy_fragment(format, opaqueAlpha).empty()) {
    Log.error("No copy to format {}", static_cast<int>(format));
  } else {
    pipeline = make_copy_pipeline(format, opaqueAlpha, color_pass(Load::DontCare));
  }
  // Kept even when it failed, so a broken copy logs once.
  g_copyPipelines.emplace(std::pair{format, opaqueAlpha}, pipeline);
  return pipeline;
}

VkPipeline present_pipeline() noexcept { return g_presentPipeline; }
} // namespace aurora::gfx::vulkan4xenon

namespace aurora::gfx::backend {
Pipeline::~Pipeline() {
  if (pipeline != VK_NULL_HANDLE) {
    vulkan4xenon::retire([pipeline = pipeline] { vkDestroyPipeline(vulkan4xenon::g_device, pipeline, nullptr); });
  }
}
} // namespace aurora::gfx::backend
