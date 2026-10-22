// Aurora's GX backend interface on vulkan4xenon, the Vulkan 1.0 subset over the Xbox 360's GPU.
#include "backend.hpp"

#include "../gfx/frame_packet.hpp"
#include "../gx/gx.hpp"
#include "../imgui.hpp"

#include <aurora/aurora.h>
#include <aurora/gfx.h>
#include <aurora/imgui.h>
#include <imgui.h>

#include <absl/container/flat_hash_map.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <iterator>
#include <mutex>

namespace aurora::gfx::vulkan4xenon {
namespace {
// Each set's textures sit at their fetch constants; a set stays cached while draws use it (as Dawn's bind groups,
// gfx/resource_cache.cpp).
constexpr uint32_t TextureSetRetainFrames = 32;
constexpr uint32_t TextureSetSweepPeriod = 16;
// A depth image this many texels or fewer on a side can't be sampled: a resolve can't write its packed mip tail
// (vulkan4xenon's docs/SUPPORT.md, Images).
constexpr uint32_t SampledDepthMinSide = 16;

struct CachedSet {
  VkDescriptorSet set = VK_NULL_HANDLE;
  uint32_t lastUsedFrame = 0;
  // The textures whose addresses make up the key, kept while the set may be drawn with.
  TextureSet textures;
};

backend::Limits g_limits{
    // vulkan4xenon's minUniformBufferOffsetAlignment, at which the GPU loads a block straight from the buffer (its
    // docs/SUPPORT.md, Descriptor sets).
    .uniformAlignment = 256,
    .storageAlignment = 16,
    // vkCmdCopyBufferToImage takes rows bufferRowLength texels long (vulkan4xenon's docs/SUPPORT.md, Commands).
    .textureRowAlignment = 4,
    .bcTextures = false,
    .cmprAsBc1 = false,
    // BC1 would need power-of-two sides; vulkan4xenon has no BC1 yet.
    .bcPowerOfTwo = true,
    // R8 and RG8 textures sample as intensity, and as intensity and alpha, through their views' component mappings
    // (vulkan4xenon's docs/SUPPORT.md, Images), so they're uploaded at 1 and 2 bytes a texel, not 4.
    .componentSwizzle = true,
    .pipelineThreads = true,
    // The vertex cache hashes each array a draw reads, whole (encoding.cpp), and takes the hashes the FIFO worker made
    // as it copied the arrays and vertex bytes, while warm, where the render worker would read them cold.
    .arraySizes = true,
    .contentHashes = true,
    // Aurora's own range fog table, which the backend copies into a texture for glsl.hpp's programs (encoding.cpp).
    .fogRangeTable = nullptr,
    // Off as on the null backend, so a pipeline's shader config is the one a run there recorded: the field changes
    // ColorChannelConfig (gx.cpp), and glsl.hpp's programs have no lighting yet.
    .unitLightTerms = false,
};

// The fragment stage, the only one that samples, and any set or pool update can come from several threads.
std::mutex g_descriptorMutex;
VkDescriptorPool g_descriptorPool = VK_NULL_HANDLE;

std::mutex g_samplerMutex;
absl::flat_hash_map<uint64_t, VkSampler> g_samplers;
// Drawn with in place of a sampler the layer refuses: the first level, filtered, clamped.
VkSampler g_fallbackSampler = VK_NULL_HANDLE;

std::mutex g_setMutex;
absl::flat_hash_map<BindGroupRef, CachedSet> g_sets;

// Bound where a draw samples nothing, as Dawn's empty texture is.
std::shared_ptr<backend::Texture> g_emptyTexture;
VkDescriptorSet g_emptySet = VK_NULL_HANDLE;

VkSamplerAddressMode address_mode(GXTexWrapMode mode) noexcept {
  switch (mode) {
  case GX_REPEAT:
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case GX_MIRROR:
    return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  default:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }
}

VkFilter filter(SamplerFilter value) noexcept {
  return value == SamplerFilter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

uint64_t sampler_key(const SamplerConfig& config) noexcept {
  Hasher hasher{};
  hasher.update(static_cast<uint32_t>(config.wrapS));
  hasher.update(static_cast<uint32_t>(config.wrapT));
  hasher.update(config.magFilter);
  hasher.update(config.minFilter);
  hasher.update(config.mipFilter);
  hasher.update(std::bit_cast<uint32_t>(config.minLod));
  hasher.update(std::bit_cast<uint32_t>(config.maxLod));
  hasher.update(config.maxAnisotropy);
  return hasher.digest();
}

void write_texture_set(VkDescriptorSet set, const TextureSet& textures) noexcept {
  std::array<VkDescriptorImageInfo, gx::MaxTextures> images{};
  for (size_t i = 0; i < images.size(); ++i) {
    const auto& handle = textures.textures[i];
    const backend::Texture* texture = handle && handle->gpu ? handle->gpu.get() : g_emptyTexture.get();
    images[i] = {
        .sampler = handle && handle->gpu ? sampler(textures.samplers[i]) : sampler({}),
        .imageView = texture->view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
  }
  const VkWriteDescriptorSet write{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set,
      .dstBinding = 0,
      .descriptorCount = static_cast<uint32_t>(images.size()),
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = images.data(),
  };
  vkUpdateDescriptorSets(g_device, 1, &write, 0, nullptr);
}
} // namespace

bool initialize_textures() noexcept {
  const VkSamplerCreateInfo fallback{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
  };
  if (!succeeded(vkCreateSampler(g_device, &fallback, nullptr, &g_fallbackSampler), "vkCreateSampler")) {
    return false;
  }
  // The layer's pools hold any number of sets, their sizes ignored (vulkan4xenon's docs/SUPPORT.md, Descriptor sets).
  const std::array sizes{
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1},
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1},
  };
  const VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets = 1,
      .poolSizeCount = sizes.size(),
      .pPoolSizes = sizes.data(),
  };
  if (!succeeded(vkCreateDescriptorPool(g_device, &poolInfo, nullptr, &g_descriptorPool), "vkCreateDescriptorPool")) {
    return false;
  }
  // Zero, as Dawn's is: its whole memory zeroed, which is the texel whatever the tiling.
  g_emptyTexture = make_texture({
      .width = 1,
      .height = 1,
      .mips = 1,
      .format = TextureFormat::RGBA8Unorm,
      .gxFormat = InvalidTextureFormat,
      .usage = backend::TextureUsage::Sampled,
      .label = "Empty texture",
  });
  if (!g_emptyTexture) {
    return false;
  }
  std::memset(g_emptyTexture->memory.mapped, 0, g_emptyTexture->memory.size);
  g_emptySet = allocate_set(texture_set_layout());
  if (g_emptySet == VK_NULL_HANDLE) {
    return false;
  }
  write_texture_set(g_emptySet, {});
  return true;
}

void shutdown_textures() noexcept {
  {
    std::lock_guard lock{g_setMutex};
    g_sets.clear();
  }
  g_emptySet = VK_NULL_HANDLE;
  g_emptyTexture.reset();
  {
    std::lock_guard lock{g_samplerMutex};
    for (const auto& [key, value] : g_samplers) {
      if (value != g_fallbackSampler) {
        vkDestroySampler(g_device, value, nullptr);
      }
    }
    g_samplers.clear();
    vkDestroySampler(g_device, g_fallbackSampler, nullptr);
    g_fallbackSampler = VK_NULL_HANDLE;
  }
  if (g_descriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(g_device, g_descriptorPool, nullptr);
    g_descriptorPool = VK_NULL_HANDLE;
  }
}

VkDescriptorSet allocate_set(VkDescriptorSetLayout layout) noexcept {
  const VkDescriptorSetAllocateInfo info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = g_descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &layout,
  };
  VkDescriptorSet set = VK_NULL_HANDLE;
  std::lock_guard lock{g_descriptorMutex};
  if (!succeeded(vkAllocateDescriptorSets(g_device, &info, &set), "vkAllocateDescriptorSets")) {
    return VK_NULL_HANDLE;
  }
  return set;
}

void free_set(VkDescriptorSet set) noexcept {
  // Destroying the pool freed every set.
  if (set == VK_NULL_HANDLE || g_descriptorPool == VK_NULL_HANDLE) {
    return;
  }
  std::lock_guard lock{g_descriptorMutex};
  succeeded(vkFreeDescriptorSets(g_device, g_descriptorPool, 1, &set), "vkFreeDescriptorSets");
}

VkSampler sampler(const SamplerConfig& config) noexcept {
  const uint64_t key = sampler_key(config);
  std::lock_guard lock{g_samplerMutex};
  if (const auto it = g_samplers.find(key); it != g_samplers.end()) {
    return it->second;
  }
  // The layer refuses a minLod above 0 and ignores maxLod (vulkan4xenon's docs/SUPPORT.md, Samplers), and GX's
  // come in sixteenths. So a texture GX keeps off its finest levels draws sharper here, and one it keeps off its
  // coarsest samples them anyway. Anisotropy as Aurora's rules give it (gfx/texture.cpp, sampler_config), at most the
  // layer's 16, which its sampler rounds down to a power of two.
  const bool linearMips = config.mipFilter == SamplerMipFilter::Linear;
  const float anisotropy = std::min(static_cast<float>(config.maxAnisotropy), max_anisotropy());
  const VkSamplerCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = filter(config.magFilter),
      .minFilter = filter(config.minFilter),
      .mipmapMode =
          config.mipFilter == SamplerMipFilter::Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = address_mode(config.wrapS),
      .addressModeV = address_mode(config.wrapT),
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .anisotropyEnable = anisotropy > 1.f,
      .maxAnisotropy = std::max(anisotropy, 1.f),
      .minLod = 0.f,
      .maxLod = linearMips ? std::floor(config.maxLod) : config.maxLod,
  };
  VkSampler value = VK_NULL_HANDLE;
  if (!succeeded(vkCreateSampler(g_device, &info, nullptr, &value), "vkCreateSampler")) {
    // Under its key all the same, so it fails once.
    Log.error("Drawing with the fallback sampler in place of one vulkan4xenon refused (filters {} {} {}, LOD {} to {}, "
              "anisotropy {})",
              static_cast<int>(config.magFilter), static_cast<int>(config.minFilter),
              static_cast<int>(config.mipFilter), config.minLod, config.maxLod, config.maxAnisotropy);
    value = g_fallbackSampler;
  }
  g_samplers.emplace(key, value);
  return value;
}

std::shared_ptr<backend::Texture> make_texture(const backend::TextureDesc& desc, VkSampleCountFlagBits samples,
                                               VkImageUsageFlags transfer, bool depthSampled) noexcept {
  using backend::TextureUsage;
  const bool depth = desc.format == TextureFormat::Depth32Float;
  // BC1 only as sampled images, where the layer has it (samples_bc1); 32-bit floats only sampled, as the layer has them
  // (vulkan4xenon's docs/SUPPORT.md, Formats).
  const bool bc1 = desc.format == TextureFormat::BC1RGBAUnorm && desc.usage == TextureUsage::Sampled;
  const bool floats = desc.format == TextureFormat::R32Float && desc.usage == TextureUsage::Sampled;
  const bool red = desc.format == TextureFormat::R8Unorm && desc.usage == TextureUsage::Sampled;
  const bool redGreen = desc.format == TextureFormat::RG8Unorm && desc.usage == TextureUsage::Sampled;
  if (!depth && !bc1 && !floats && !red && !redGreen && desc.format != TextureFormat::RGBA8Unorm) {
    Log.error("No texture for {}: format {} isn't RGBA8, swizzled BC1, or sampled R32Float, R8 or RG8", desc.label,
              static_cast<uint32_t>(desc.format));
    return {};
  }
  auto texture = std::make_shared<backend::Texture>();
  texture->width = desc.width;
  texture->height = desc.height;
  texture->bc1 = bc1;
  texture->texelBytes = red ? 1 : redGreen ? 2 : 4;
  VkImageCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = depth      ? VK_FORMAT_D24_UNORM_S8_UINT
                : bc1      ? VK_FORMAT_BC1_RGBA_UNORM_BLOCK
                : floats   ? VK_FORMAT_R32_SFLOAT
                : red      ? VK_FORMAT_R8_UNORM
                : redGreen ? VK_FORMAT_R8G8_UNORM
                           : VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {desc.width, desc.height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = samples,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  switch (desc.usage) {
  case TextureUsage::Sampled:
    info.mipLevels = desc.mips;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    break;
  case TextureUsage::Copy:
  case TextureUsage::Conv:
  case TextureUsage::Render:
    info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    break;
  case TextureUsage::Depth:
    // Sampled too, for depth copies and GXPeekZ, where the layer can resolve into it: more than 16 texels on a side
    // (vulkan4xenon's docs/SUPPORT.md, Images).
    info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                 (depthSampled && desc.width > SampledDepthMinSide && desc.height > SampledDepthMinSide
                      ? VK_IMAGE_USAGE_SAMPLED_BIT
                      : 0);
    break;
  }
  // The layer samples no multisampled image (vulkan4xenon's docs/SUPPORT.md, Images).
  if (samples != VK_SAMPLE_COUNT_1_BIT) {
    info.usage &= ~VK_IMAGE_USAGE_SAMPLED_BIT;
  }
  info.usage |= transfer;
  if (!succeeded(vkCreateImage(g_device, &info, nullptr, &texture->image), "vkCreateImage")) {
    Log.error("No image for {} ({}x{})", desc.label, desc.width, desc.height);
    return {};
  }
  VkMemoryRequirements requirements;
  vkGetImageMemoryRequirements(g_device, texture->image, &requirements);
  if (!allocate(requirements, texture->memory, desc.ui) ||
      !succeeded(vkBindImageMemory(g_device, texture->image, texture->memory.memory, texture->memory.offset),
                 "vkBindImageMemory")) {
    Log.error("No memory for {} ({}x{})", desc.label, desc.width, desc.height);
    return {};
  }
  // R8 is intensity, read into every channel, as the WebGPU backend's swizzle reads R8_PC and I8; RG8 is intensity
  // and alpha, as it reads RG8_PC.
  const VkComponentSwizzle r = VK_COMPONENT_SWIZZLE_R;
  const VkComponentMapping components = red        ? VkComponentMapping{r, r, r, r}
                                        : redGreen ? VkComponentMapping{r, r, r, VK_COMPONENT_SWIZZLE_G}
                                                   : VkComponentMapping{};
  // A depth target's attachment view has depth and stencil, and its sampled one depth alone.
  VkImageViewCreateInfo viewInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = texture->image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = info.format,
      .components = components,
      .subresourceRange = {depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                           0, info.mipLevels, 0, 1},
  };
  if (!succeeded(vkCreateImageView(g_device, &viewInfo, nullptr, &texture->view), "vkCreateImageView")) {
    return {};
  }
  if (depth && (info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0) {
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (!succeeded(vkCreateImageView(g_device, &viewInfo, nullptr, &texture->depthView), "vkCreateImageView")) {
      return {};
    }
  }
  return texture;
}

VkDescriptorSet texture_set(BindGroupRef ref) noexcept {
  if (ref == 0) {
    return g_emptySet;
  }
  std::lock_guard lock{g_setMutex};
  const auto it = g_sets.find(ref);
  if (it == g_sets.end() || it->second.set == VK_NULL_HANDLE) {
    Log.error("No texture set {:x}", ref);
    return g_emptySet;
  }
  return it->second.set;
}

void expire_texture_sets() noexcept {
  const uint32_t frame = current_frame();
  if (frame == UINT32_MAX || frame % TextureSetSweepPeriod != 0) {
    return;
  }
  std::lock_guard lock{g_setMutex};
  for (auto it = g_sets.begin(); it != g_sets.end();) {
    if (frame - it->second.lastUsedFrame > TextureSetRetainFrames) {
      retire([set = it->second.set, textures = std::move(it->second.textures)] { free_set(set); });
      g_sets.erase(it++);
    } else {
      ++it;
    }
  }
}
} // namespace aurora::gfx::vulkan4xenon

namespace aurora::gfx::backend {
using namespace vulkan4xenon;

Texture::~Texture() {
  if (g_device == VK_NULL_HANDLE) {
    return;
  }
  retire([image = image, view = view, depthView = depthView, memory = memory, framebuffer = framebuffer,
          sourceSets = sourceSets] {
    vkDestroyFramebuffer(g_device, framebuffer, nullptr);
    for (const auto set : sourceSets) {
      free_set(set);
    }
    vkDestroyImageView(g_device, depthView, nullptr);
    vkDestroyImageView(g_device, view, nullptr);
    vkDestroyImage(g_device, image, nullptr);
    free_allocation(memory);
  });
}

UploadBuffer::~UploadBuffer() {
  if (g_device == VK_NULL_HANDLE) {
    return;
  }
  retire([buffer = buffer, memory = memory] {
    vkDestroyBuffer(g_device, buffer, nullptr);
    free_allocation(memory);
  });
}

bool initialize(AuroraBackend& backend) noexcept {
  if (!initialize_device()) {
    return false;
  }
  log_memory_step("vulkan4xenon's device");
  // CMPR without transparent texels goes to the GPU as BC1, an eighth of RGBA8's size, where the layer has it.
  g_limits.cmprAsBc1 = samples_bc1();
  Log.info("CMPR textures {}", g_limits.cmprAsBc1 ? "as BC1 where opaque" : "as RGBA8: vulkan4xenon has no BC1");
  if (!initialize_pipelines()) {
    shutdown_device();
    return false;
  }
  log_memory_step("the pipelines and the shader pack");
  if (!initialize_textures()) {
    shutdown();
    return false;
  }
  log_memory_step("the textures");
  if (!initialize_frames()) {
    shutdown();
    return false;
  }
  log_memory_step("the frames");
  backend = BACKEND_VULKAN;
  Log.info("vulkan4xenon backend: {}x{} EFB", efb_size().width, efb_size().height);
  return true;
}

void shutdown() noexcept {
  // The frames' buffers go at once, so the GPU must be done with them.
  if (g_device != VK_NULL_HANDLE) {
    succeeded(vkDeviceWaitIdle(g_device), "vkDeviceWaitIdle");
  }
  shutdown_ui();
  shutdown_frames();
  shutdown_textures();
  shutdown_pipelines();
  shutdown_device();
}

bool uses_sdl_window() noexcept { return false; }

bool can_present() noexcept { return true; }

void resize(const AuroraWindowSize& size) noexcept { (void)size; }

void refresh_surface() noexcept {}

// Every texture tiled, whatever its size (vulkan4xenon's docs/SUPPORT.md, Images).
std::shared_ptr<Texture> create_texture(const TextureDesc& desc) noexcept { return make_texture(desc); }

#ifdef AURORA_DIAGNOSTICS
size_t texture_bytes(const Texture& texture) noexcept { return texture.memory.size; }
#endif

std::shared_ptr<UploadBuffer> stage_upload(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage,
                                           bool ui) noexcept {
  const uint32_t pitch = AURORA_ALIGN(bytesPerRow, limits().textureRowAlignment);
  auto upload = std::make_shared<UploadBuffer>();
  const VkBufferCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = static_cast<VkDeviceSize>(pitch) * rowsPerImage,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  if (!succeeded(vkCreateBuffer(g_device, &info, nullptr, &upload->buffer), "vkCreateBuffer")) {
    return {};
  }
  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(g_device, upload->buffer, &requirements);
  if (!allocate(requirements, upload->memory, ui) ||
      !succeeded(vkBindBufferMemory(g_device, upload->buffer, upload->memory.memory, upload->memory.offset),
                 "vkBindBufferMemory")) {
    Log.error("No memory for an upload of {} bytes past the frame's pool", info.size);
    return {};
  }
  for (uint32_t row = 0; row < rowsPerImage; ++row) {
    std::memcpy(upload->memory.mapped + static_cast<size_t>(row) * pitch, data + static_cast<size_t>(row) * bytesPerRow,
                bytesPerRow);
  }
  return upload;
}

// One pipeline a config, never a prepass: what Dawn's backend draws in two passes, destination alpha, make_pipeline
// refuses. Every pass draws into one RGBA8 colour target, with or without depth, at one sample or, the EFB's, at its
// samples (efb_layout and the offscreen targets make_texture creates).
CompiledPipeline create_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout) noexcept {
  if (layout.colorAttachmentCount != 1 ||
      (layout.sampleCount != 1 && layout.sampleCount != static_cast<uint32_t>(efb_samples()))) {
    Log.error("No pipeline for a pass of {} colour targets at {} samples; its draws are skipped",
              layout.colorAttachmentCount, layout.sampleCount);
    return {};
  }
  return {.main = make_pipeline(config, static_cast<VkSampleCountFlagBits>(layout.sampleCount))};
}

std::shared_ptr<Pipeline> create_pipeline(const clear::PipelineConfig& config,
                                          const RenderTargetLayout& layout) noexcept {
  return make_pipeline(config, layout);
}

void pipelines_loaded() noexcept {}

void prepare_textures(BindGroupRef ref, const TextureSet& set) noexcept {
  std::lock_guard lock{g_setMutex};
  if (const auto it = g_sets.find(ref); it != g_sets.end()) {
    it->second.lastUsedFrame = current_frame();
    return;
  }
  const VkDescriptorSet descriptorSet = allocate_set(texture_set_layout());
  if (descriptorSet != VK_NULL_HANDLE) {
    write_texture_set(descriptorSet, set);
  }
  g_sets.emplace(ref, CachedSet{.set = descriptorSet, .lastUsedFrame = current_frame(), .textures = set});
}

bool converts_copy(GXTexFmt format) noexcept { return converts(format); }

const Limits& limits() noexcept { return g_limits; }

RenderTargetLayout efb_layout() noexcept {
  const auto size = efb_size();
  RenderTargetLayout layout{
      .colorAttachmentCount = 1,
      .colorAttachments = {{{
          .semantic = ColorAttachmentSemantic::SceneColor,
          .format = TextureFormat::RGBA8Unorm,
          .width = size.width,
          .height = size.height,
      }}},
      .depthStencilFormat = TextureFormat::Depth32Float,
      .sampleCount = static_cast<uint32_t>(efb_samples()),
  };
  detail::finalize_render_target_layout(layout);
  return layout;
}
} // namespace aurora::gfx::backend

namespace aurora::imgui {
void initialize() noexcept {}

void shutdown() noexcept { ImGui::DestroyContext(); }

void process_event(const SDL_Event& event) noexcept { (void)event; }

// new_frame is ui.cpp's.
} // namespace aurora::imgui

// The swapchain presents FIFO, always.
void aurora_enable_vsync(bool enabled) { (void)enabled; }

bool aurora_present_waits_for_vblank() { return true; }

void aurora_set_resampler(AuroraSampler sampler) { (void)sampler; }

const AuroraBackend* aurora_get_available_backends(size_t* count) {
  static constexpr AuroraBackend Backends[]{BACKEND_VULKAN};
  if (count != nullptr) {
    *count = std::size(Backends);
  }
  return Backends;
}

void aurora_gpu_frame_time(uint64_t* lastNs, uint64_t* meanNs, uint64_t* maxNs, uint64_t* count) {
  // No timer queries in vulkan4xenon, so no samples.
  for (uint64_t* out : {lastNs, meanNs, maxNs, count}) {
    if (out != nullptr) {
      *out = 0;
    }
  }
}

// aurora_imgui_add_texture is ui.cpp's.
