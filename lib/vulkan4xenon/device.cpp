// vulkan4xenon's instance, device, swapchain and memory, and the submission of each frame.
#include "backend.hpp"

#include "../thread.hpp"

#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace aurora::gfx::vulkan4xenon {
VkDevice g_device = VK_NULL_HANDLE;
VkQueue g_queue = VK_NULL_HANDLE;

namespace {
// Allocations share blocks: each vkAllocateMemory is xsdk_gpu_alloc's physical memory in 64 KiB pages, at most 4096
// (vulkan4xenon's docs/SUPPORT.md, Memory and buffers). The GPU's memory type is both the first pools': the first in
// 8 MiB blocks while it can, then exact ones, the second in exact ones. The third takes the layer's cached type, which
// it offers only to buffers that are only copied from: copies run on the CPU, which reads cached memory at the cache's
// speed, where the GPU's is read uncached.
constexpr VkDeviceSize MemoryBlockSize = 8 * 1024 * 1024;
constexpr VkDeviceSize VideoRamGranularity = 64 * 1024;
constexpr VkDeviceSize MainRamGranularity = 64 * 1024;
constexpr VkDeviceSize CachedGranularity = 4 * 1024;
constexpr uint32_t VideoPool = 0;
constexpr uint32_t MainPool = 1;
constexpr uint32_t CachedPool = 2;
// The waiter only waits: 64 KiB, not a pthread's default.
constexpr size_t WaiterStackSize = 64 * 1024;

struct MemoryBlock {
  VkDeviceMemory memory = VK_NULL_HANDLE;
  uint8_t* mapped = nullptr;
  VkDeviceSize size = 0;
  // Offset to size of each free range.
  std::map<VkDeviceSize, VkDeviceSize> free;

  bool empty() const noexcept { return free.size() == 1 && free.begin()->second == size; }
};

struct MemoryPool {
  const char* name;
  VkDeviceSize granularity;
  bool wholeBlocksFirst = true;
  uint32_t type = UINT32_MAX;
  bool wholeBlocks = true;
  // The smallest block the layer refused since the pool last gave memory back, or 0: nothing that big is asked for
  // until then, since each refusal is logged here.
  VkDeviceSize refused = 0;
  std::vector<MemoryBlock> blocks;
#ifdef AURORA_DIAGNOSTICS
  // Everything taken from the layer, blocks and dedicated buffers, and the part of it in use.
  VkDeviceSize taken = 0;
  VkDeviceSize takenPeak = 0;
  VkDeviceSize used = 0;
  VkDeviceSize usedPeak = 0;
#endif
};

struct PendingFrame {
  VkFence fence;
  uint64_t serial;
  size_t slot;
#ifdef AURORA_DIAGNOSTICS
  uint64_t frameId;
  std::chrono::steady_clock::time_point submitted;
#endif
};

struct Retired {
  uint64_t serial;
  std::function<void()> release;
};

VkInstance g_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_physicalDevice = VK_NULL_HANDLE;
VkSurfaceKHR g_surface = VK_NULL_HANDLE;
VkSwapchainKHR g_swapchain = VK_NULL_HANDLE;
VkExtent2D g_swapchainExtent{};
std::vector<VkImage> g_swapchainImages;
std::vector<VkImageView> g_swapchainViews;
std::vector<VkFramebuffer> g_swapchainFramebuffers;
// By how depth is left (DepthStore), whether a loaded depth was left to be read, then colour load op, then depth load
// op. Scenes only load or clear.
std::array<std::array<std::array<std::array<VkRenderPass, 3>, 3>, 2>, 3> g_scenePasses{};
std::array<VkRenderPass, 3> g_colorPasses{};
// The swapchain's, cleared and left as it is, by Load.
std::array<VkRenderPass, 3> g_presentPasses{};

std::mutex g_memoryMutex;
std::array<MemoryPool, 3> g_pools{
    MemoryPool{.name = "memory in whole blocks", .granularity = VideoRamGranularity},
    MemoryPool{.name = "memory in exact blocks",
               .granularity = MainRamGranularity,
               .wholeBlocksFirst = false,
               .wholeBlocks = false},
    MemoryPool{.name = "cached memory", .granularity = CachedGranularity},
};

// Whether the cached pool can hold what has these memory type bits: a type of its own the requirements list.
bool fits_cached(uint32_t memoryTypeBits) noexcept {
  const uint32_t type = g_pools[CachedPool].type;
  return type != g_pools[VideoPool].type && (memoryTypeBits & (1u << type)) != 0;
}

// The most anisotropy samplers may ask for: the layer's limit when it has the samplerAnisotropy feature, else 1.
float g_maxAnisotropy = 1.f;

VkCommandPool g_commandPool = VK_NULL_HANDLE;
std::array<VkCommandBuffer, detail::FrameSlotCount> g_commands{};
std::array<VkFence, detail::FrameSlotCount> g_fences{};
// A slot's token: taken by begin_frame, given back once the GPU has finished the slot's frame. Posted once each when
// made.
std::array<Semaphore, detail::FrameSlotCount> g_slotFree;

// Frames are numbered as they are submitted. The waiter thread marks each one complete when its fence signals.
// Semaphores carry every wakeup, each with its own condition variable: with one shared by the waiter and begin_frame,
// the game thread can sleep on with its frame complete.
std::atomic<uint64_t> g_submitted = 0;
std::atomic<uint64_t> g_completed = 0;
std::mutex g_pendingMutex;
std::deque<PendingFrame> g_pending;
// Counts g_pending's frames, plus one to stop the waiter, which finds no frame then.
Semaphore g_pendingCount;
pthread_t g_waiter;
bool g_waiterStarted = false;

std::mutex g_retiredMutex;
std::deque<Retired> g_retired;

VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment) noexcept {
  return (value + alignment - 1) / alignment * alignment;
}

#ifdef AURORA_DIAGNOSTICS
// The GPU's frame times, from the waiter thread: see wait_for_frames.
std::atomic<uint64_t> g_gpuBusyNs = 0;

void count_taken(MemoryPool& pool, VkDeviceSize size) noexcept {
  pool.taken += size;
  pool.takenPeak = std::max(pool.takenPeak, pool.taken);
}

void count_used(MemoryPool& pool, VkDeviceSize size) noexcept {
  pool.used += size;
  pool.usedPeak = std::max(pool.usedPeak, pool.used);
}
#endif

// The layer has memory again: the pool may ask for whole blocks, and for sizes it refused.
void given_back(MemoryPool& pool) noexcept {
  pool.wholeBlocks = pool.wholeBlocksFirst;
  pool.refused = 0;
}

bool take_range(MemoryBlock& block, VkDeviceSize size, VkDeviceSize alignment, VkDeviceSize& offset) noexcept {
  for (auto it = block.free.begin(); it != block.free.end(); ++it) {
    const VkDeviceSize start = it->first;
    const VkDeviceSize end = it->first + it->second;
    const VkDeviceSize aligned = align_up(start, alignment);
    if (aligned + size > end) {
      continue;
    }
    block.free.erase(it);
    if (aligned > start) {
      block.free.emplace(start, aligned - start);
    }
    if (aligned + size < end) {
      block.free.emplace(aligned + size, end - aligned - size);
    }
    offset = aligned;
    return true;
  }
  return false;
}

bool add_block(MemoryPool& pool, VkDeviceSize minSize) noexcept {
  const VkDeviceSize exactSize = align_up(minSize, pool.granularity);
  if (pool.refused != 0 && exactSize >= pool.refused) {
    return false;
  }
  VkDeviceSize size = pool.wholeBlocks ? std::max(MemoryBlockSize, exactSize) : exactSize;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkResult result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
  while (true) {
    const VkMemoryAllocateInfo info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = size,
        .memoryTypeIndex = pool.type,
    };
    result = vkAllocateMemory(g_device, &info, nullptr, &memory);
    if (result == VK_SUCCESS || size == exactSize) {
      break;
    }
    Log.info("vkAllocateMemory: {} has no {} bytes left ({}); its blocks are now what each allocation needs", pool.name,
             size, static_cast<int>(result));
    pool.wholeBlocks = false;
    size = exactSize;
  }
  if (result != VK_SUCCESS) {
    // The first pool running out is expected; the second takes over.
    Log.warn("vkAllocateMemory: {} bytes of {}: {}", size, pool.name, static_cast<int>(result));
    pool.refused = size;
    return false;
  }
  void* mapped = nullptr;
  if (!succeeded(vkMapMemory(g_device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory")) {
    vkFreeMemory(g_device, memory, nullptr);
    return false;
  }
  pool.blocks.push_back({
      .memory = memory,
      .mapped = static_cast<uint8_t*>(mapped),
      .size = size,
      .free = {{0, size}},
  });
#ifdef AURORA_DIAGNOSTICS
  count_taken(pool, size);
#endif
  return true;
}

bool create_render_pass(bool depth, Load colorLoad, Load depthLoad, DepthStore depthStore, bool fromRead,
                        VkRenderPass& out, VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM) noexcept {
  const auto load_op = [](Load load) {
    switch (load) {
    case Load::Load:
      return VK_ATTACHMENT_LOAD_OP_LOAD;
    case Load::Clear:
      return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case Load::DontCare:
      break;
    }
    return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  };
  // The layer replays in order and its barriers record nothing (its docs/SUPPORT.md, Commands), so colour's layouts
  // change nothing, and GENERAL suits every use. Depth's say when a shader reads it next: the layer copies it out of
  // eDRAM into its image only for a pass that stores it and leaves it to be read (its docs/SUPPORT.md, Images), so
  // passes the next one loads keep it as an attachment.
  const VkImageLayout depthBefore = depthLoad != Load::Load ? VK_IMAGE_LAYOUT_UNDEFINED
                                    : fromRead              ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                            : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  const VkImageLayout depthAfter = depthStore == DepthStore::Read ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                                  : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  const std::array attachments{
      VkAttachmentDescription{
          .format = colorFormat,
          .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = load_op(colorLoad),
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
          .finalLayout = VK_IMAGE_LAYOUT_GENERAL,
      },
      VkAttachmentDescription{
          .format = VK_FORMAT_D24_UNORM_S8_UINT,
          .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = load_op(depthLoad),
          .storeOp = depthStore == DepthStore::Dropped ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                                                       : VK_ATTACHMENT_STORE_OP_STORE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = depthBefore,
          .finalLayout = depthAfter,
      },
  };
  const VkAttachmentReference colorRef{.attachment = 0, .layout = VK_IMAGE_LAYOUT_GENERAL};
  const VkAttachmentReference depthRef{.attachment = 1, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
  const VkSubpassDescription subpass{
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorRef,
      .pDepthStencilAttachment = depth ? &depthRef : nullptr,
  };
  const VkRenderPassCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = depth ? 2u : 1u,
      .pAttachments = attachments.data(),
      .subpassCount = 1,
      .pSubpasses = &subpass,
  };
  return succeeded(vkCreateRenderPass(g_device, &info, nullptr, &out), "vkCreateRenderPass");
}

bool create_render_passes() noexcept {
  constexpr std::array SceneLoads{Load::Load, Load::Clear};
  for (const DepthStore store : {DepthStore::Kept, DepthStore::Read, DepthStore::Dropped}) {
    for (const bool fromRead : {false, true}) {
      for (const Load color : SceneLoads) {
        for (const Load depth : SceneLoads) {
          if (!create_render_pass(true, color, depth, store, fromRead,
                                  g_scenePasses[static_cast<size_t>(store)][fromRead][static_cast<size_t>(color)]
                                               [static_cast<size_t>(depth)])) {
            return false;
          }
        }
      }
    }
  }
  for (const Load color : {Load::Load, Load::Clear, Load::DontCare}) {
    if (!create_render_pass(false, color, Load::DontCare, DepthStore::Kept, false,
                            g_colorPasses[static_cast<size_t>(color)])) {
      return false;
    }
  }
  for (const Load color : {Load::Clear, Load::DontCare}) {
    if (!create_render_pass(false, color, Load::DontCare, DepthStore::Kept, false,
                            g_presentPasses[static_cast<size_t>(color)], VK_FORMAT_B8G8R8A8_UNORM)) {
      return false;
    }
  }
  return true;
}

bool create_instance() noexcept {
  const std::array extensions{VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_DISPLAY_EXTENSION_NAME};
  const VkApplicationInfo app{
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = g_config.appName,
      .pEngineName = "Aurora",
      .apiVersion = VK_API_VERSION_1_0,
  };
  const VkInstanceCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
      .enabledExtensionCount = extensions.size(),
      .ppEnabledExtensionNames = extensions.data(),
  };
  if (!succeeded(vkCreateInstance(&info, nullptr, &g_instance), "vkCreateInstance")) {
    return false;
  }
  uint32_t count = 1;
  const VkResult result = vkEnumeratePhysicalDevices(g_instance, &count, &g_physicalDevice);
  if ((result != VK_SUCCESS && result != VK_INCOMPLETE) || count == 0) {
    Log.error("vkEnumeratePhysicalDevices: {}, {} devices", static_cast<int>(result), count);
    return false;
  }
  return true;
}

bool find_memory_types() noexcept {
  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(g_physicalDevice, &properties);
  // The GPU's type, device-local and host-visible, for the first pools, then the cached one, host-visible and cached,
  // for the third (its docs/SUPPORT.md, Memory and buffers); without it, the third is the first's and never taken.
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    if ((properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
      g_pools[VideoPool].type = i;
      g_pools[MainPool].type = i;
      g_pools[CachedPool].type = i;
      break;
    }
  }
  if (g_pools[VideoPool].type == UINT32_MAX) {
    Log.error("No host-visible memory type");
    return false;
  }
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    if ((properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) {
      g_pools[CachedPool].type = i;
      break;
    }
  }
  return true;
}

bool create_surface() noexcept {
  uint32_t count = 1;
  VkDisplayPropertiesKHR display;
  if (!succeeded(vkGetPhysicalDeviceDisplayPropertiesKHR(g_physicalDevice, &count, &display),
                 "vkGetPhysicalDeviceDisplayPropertiesKHR")) {
    return false;
  }
  count = 1;
  VkDisplayModePropertiesKHR mode;
  if (!succeeded(vkGetDisplayModePropertiesKHR(g_physicalDevice, display.display, &count, &mode),
                 "vkGetDisplayModePropertiesKHR")) {
    return false;
  }
  const VkDisplaySurfaceCreateInfoKHR info{
      .sType = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
      .displayMode = mode.displayMode,
      .transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .globalAlpha = 1.f,
      .alphaMode = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR,
      .imageExtent = mode.parameters.visibleRegion,
  };
  return succeeded(vkCreateDisplayPlaneSurfaceKHR(g_instance, &info, nullptr, &g_surface),
                   "vkCreateDisplayPlaneSurfaceKHR");
}

bool create_device() noexcept {
  const float priority = 1.f;
  const VkDeviceQueueCreateInfo queue{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
  };
  const std::array extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  // Sampler anisotropy, where the layer has it, for GX's GX_ANISO_2 and GX_ANISO_4 (gfx/texture.cpp, anisotropy).
  VkPhysicalDeviceFeatures supported{};
  vkGetPhysicalDeviceFeatures(g_physicalDevice, &supported);
  const VkPhysicalDeviceFeatures features{.samplerAnisotropy = supported.samplerAnisotropy};
  const VkDeviceCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue,
      .enabledExtensionCount = extensions.size(),
      .ppEnabledExtensionNames = extensions.data(),
      .pEnabledFeatures = &features,
  };
  if (!succeeded(vkCreateDevice(g_physicalDevice, &info, nullptr, &g_device), "vkCreateDevice")) {
    return false;
  }
  if (supported.samplerAnisotropy) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(g_physicalDevice, &properties);
    g_maxAnisotropy = properties.limits.maxSamplerAnisotropy;
  }
  vkGetDeviceQueue(g_device, 0, 0, &g_queue);
  return true;
}

bool create_swapchain() noexcept {
  VkSurfaceCapabilitiesKHR capabilities;
  if (!succeeded(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_physicalDevice, g_surface, &capabilities),
                 "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")) {
    return false;
  }
  g_swapchainExtent = capabilities.currentExtent;
  // FIFO: the display shows each frame from a vertical blank. Two images: frames take longer than a refresh.
  const VkSwapchainCreateInfoKHR info{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = g_surface,
      .minImageCount = 2,
      .imageFormat = VK_FORMAT_B8G8R8A8_UNORM,
      .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
      .imageExtent = g_swapchainExtent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = VK_PRESENT_MODE_FIFO_KHR,
      .clipped = VK_TRUE,
  };
  const VkResult created = vkCreateSwapchainKHR(g_device, &info, nullptr, &g_swapchain);
  if (created == VK_ERROR_INITIALIZATION_FAILED) {
    // No display controller, as in Xenia (vulkan4xenon's docs/SUPPORT.md, Display and swapchain): frames are drawn
    // and not shown.
    Log.warn("No swapchain: frames are drawn but not shown");
    g_swapchain = VK_NULL_HANDLE;
    return true;
  }
  if (!succeeded(created, "vkCreateSwapchainKHR")) {
    return false;
  }
  uint32_t count = 0;
  if (!succeeded(vkGetSwapchainImagesKHR(g_device, g_swapchain, &count, nullptr), "vkGetSwapchainImagesKHR")) {
    return false;
  }
  g_swapchainImages.resize(count);
  if (!succeeded(vkGetSwapchainImagesKHR(g_device, g_swapchain, &count, g_swapchainImages.data()),
                 "vkGetSwapchainImagesKHR")) {
    return false;
  }
  for (const VkImage image : g_swapchainImages) {
    const VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageView view = VK_NULL_HANDLE;
    if (!succeeded(vkCreateImageView(g_device, &viewInfo, nullptr, &view), "vkCreateImageView")) {
      return false;
    }
    g_swapchainViews.push_back(view);
    const VkFramebufferCreateInfo framebufferInfo{
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = present_pass(Load::DontCare),
        .attachmentCount = 1,
        .pAttachments = &view,
        .width = g_swapchainExtent.width,
        .height = g_swapchainExtent.height,
        .layers = 1,
    };
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    if (!succeeded(vkCreateFramebuffer(g_device, &framebufferInfo, nullptr, &framebuffer), "vkCreateFramebuffer")) {
      return false;
    }
    g_swapchainFramebuffers.push_back(framebuffer);
  }
  return true;
}

bool create_frame_sync() noexcept {
  const VkCommandPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = 0,
  };
  if (!succeeded(vkCreateCommandPool(g_device, &poolInfo, nullptr, &g_commandPool), "vkCreateCommandPool")) {
    return false;
  }
  const VkCommandBufferAllocateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = g_commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = static_cast<uint32_t>(g_commands.size()),
  };
  if (!succeeded(vkAllocateCommandBuffers(g_device, &bufferInfo, g_commands.data()), "vkAllocateCommandBuffers")) {
    return false;
  }
  const VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  for (auto& fence : g_fences) {
    if (!succeeded(vkCreateFence(g_device, &fenceInfo, nullptr, &fence), "vkCreateFence")) {
      return false;
    }
  }
  return true;
}

// Marks each submitted frame complete once its fence signals, in order, and gives its slot back. With diagnostics it
// also times each frame on the GPU without queries: from the later of its submission and the previous fence to its own
// fence. That counts this thread's wake-up too, so
// it is at least what the GPU spent.
void* wait_for_frames(void* arg) noexcept {
  (void)arg;
  thread::set_current({.name = "Aurora GPU waiter", .priority = thread::Priority::High});
#ifdef AURORA_DIAGNOSTICS
  std::chrono::steady_clock::time_point previousDone{};
#endif
  while (true) {
    g_pendingCount.wait();
    PendingFrame frame;
    {
      std::lock_guard lock{g_pendingMutex};
      if (g_pending.empty()) {
        return nullptr;
      }
      frame = g_pending.front();
      g_pending.pop_front();
    }
    succeeded(vkWaitForFences(g_device, 1, &frame.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
#ifdef AURORA_DIAGNOSTICS
    const auto done = std::chrono::steady_clock::now();
    const auto start = std::max(frame.submitted, previousDone);
    previousDone = done;
    g_gpuBusyNs.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(done - start).count(),
                          std::memory_order_relaxed);
    note_frame_path(frame.frameId, FramePath::FenceSignalled, done);
#endif
    g_completed.store(frame.serial, std::memory_order_release);
    g_slotFree[frame.slot].post();
  }
}

// Every slot starts free.
bool create_semaphores() noexcept {
  for (auto& slot : g_slotFree) {
    slot.post();
  }
  return true;
}

bool start_waiter() noexcept {
  pthread_attr_t attr;
  int ret = pthread_attr_init(&attr);
  if (ret != 0) {
    Log.error("pthread_attr_init (GPU waiter): {}", strerror(ret));
    return false;
  }
  const char* call = "pthread_attr_setstacksize";
  ret = pthread_attr_setstacksize(&attr, WaiterStackSize);
  if (ret == 0) {
    call = "pthread_create";
    ret = pthread_create(&g_waiter, &attr, wait_for_frames, nullptr);
  }
  if (const int destroyed = pthread_attr_destroy(&attr); destroyed != 0) {
    Log.error("pthread_attr_destroy (GPU waiter): {}", strerror(destroyed));
  }
  if (ret != 0) {
    Log.error("{} (GPU waiter): {}", call, strerror(ret));
    return false;
  }
  g_waiterStarted = true;
  return true;
}

// True when it ran anything.
bool run_retired(uint64_t completed) noexcept {
  std::deque<Retired> ready;
  {
    std::lock_guard lock{g_retiredMutex};
    while (!g_retired.empty() && g_retired.front().serial <= completed) {
      ready.push_back(std::move(g_retired.front()));
      g_retired.pop_front();
    }
  }
  for (auto& item : ready) {
    item.release();
  }
  return !ready.empty();
}
} // namespace

bool succeeded(VkResult result, const char* call) noexcept {
  if (result == VK_SUCCESS) {
    return true;
  }
  Log.error("{}: {}", call, static_cast<int>(result));
  return false;
}

bool initialize_device() noexcept {
  if (!create_instance() || !find_memory_types() || !create_surface() || !create_device()) {
    shutdown_device();
    return false;
  }
  if (!create_render_passes() || !create_swapchain() || !create_frame_sync() || !create_semaphores()) {
    shutdown_device();
    return false;
  }
  if (!start_waiter()) {
    shutdown_device();
    return false;
  }
  return true;
}

void shutdown_device() noexcept {
  if (g_device != VK_NULL_HANDLE) {
    succeeded(vkDeviceWaitIdle(g_device), "vkDeviceWaitIdle");
  }
  if (g_waiterStarted) {
    g_pendingCount.post();
    const int ret = pthread_join(g_waiter, nullptr);
    if (ret != 0) {
      Log.error("pthread_join (GPU waiter): {}", strerror(ret));
    }
    g_waiterStarted = false;
  }
  // A release can retire more, as a texture set's does its textures.
  while (run_retired(UINT64_MAX)) {}
  for (const auto fence : g_fences) {
    vkDestroyFence(g_device, fence, nullptr);
  }
  g_fences.fill(VK_NULL_HANDLE);
  if (g_commandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(g_device, g_commandPool, nullptr);
    g_commandPool = VK_NULL_HANDLE;
  }
  for (const auto framebuffer : g_swapchainFramebuffers) {
    vkDestroyFramebuffer(g_device, framebuffer, nullptr);
  }
  g_swapchainFramebuffers.clear();
  for (const auto view : g_swapchainViews) {
    vkDestroyImageView(g_device, view, nullptr);
  }
  g_swapchainViews.clear();
  g_swapchainImages.clear();
  if (g_swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(g_device, g_swapchain, nullptr);
    g_swapchain = VK_NULL_HANDLE;
  }
  for (auto& stored : g_scenePasses) {
    for (auto& loaded : stored) {
      for (auto& passes : loaded) {
        for (auto& pass : passes) {
          vkDestroyRenderPass(g_device, pass, nullptr);
          pass = VK_NULL_HANDLE;
        }
      }
    }
  }
  for (auto& pass : g_colorPasses) {
    vkDestroyRenderPass(g_device, pass, nullptr);
    pass = VK_NULL_HANDLE;
  }
  for (auto& pass : g_presentPasses) {
    vkDestroyRenderPass(g_device, pass, nullptr);
    pass = VK_NULL_HANDLE;
  }
  for (auto& pool : g_pools) {
    for (const auto& block : pool.blocks) {
      vkFreeMemory(g_device, block.memory, nullptr);
    }
    pool.blocks.clear();
  }
  if (g_device != VK_NULL_HANDLE) {
    vkDestroyDevice(g_device, nullptr);
    g_device = VK_NULL_HANDLE;
    g_queue = VK_NULL_HANDLE;
  }
  if (g_surface != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(g_instance, g_surface, nullptr);
    g_surface = VK_NULL_HANDLE;
  }
  if (g_instance != VK_NULL_HANDLE) {
    vkDestroyInstance(g_instance, nullptr);
    g_instance = VK_NULL_HANDLE;
  }
}

// A buffer can share a page with an image: the layer's bufferImageGranularity is 1 (its docs/SUPPORT.md, Physical
// device).
bool allocate(const VkMemoryRequirements& requirements, Allocation& out) noexcept {
  std::lock_guard lock{g_memoryMutex};
  for (uint32_t poolIndex : {CachedPool, VideoPool, MainPool}) {
    auto& pool = g_pools[poolIndex];
    if ((requirements.memoryTypeBits & (1u << pool.type)) == 0 ||
        (poolIndex == CachedPool && !fits_cached(requirements.memoryTypeBits))) {
      continue;
    }
    for (size_t attempt = 0; attempt < 2; ++attempt) {
      for (auto& block : pool.blocks) {
        VkDeviceSize offset = 0;
        if (take_range(block, requirements.size, requirements.alignment, offset)) {
#ifdef AURORA_DIAGNOSTICS
          count_used(pool, requirements.size);
#endif
          out = {
              .memory = block.memory,
              .offset = offset,
              .size = requirements.size,
              .mapped = block.mapped + offset,
              .pool = poolIndex,
          };
          return true;
        }
      }
      if (attempt == 0 && !add_block(pool, requirements.size)) {
        break;
      }
    }
  }
  Log.error("No memory for {} bytes", requirements.size);
  return false;
}

void free_allocation(const Allocation& allocation) noexcept {
  if (allocation.memory == VK_NULL_HANDLE) {
    return;
  }
  std::lock_guard lock{g_memoryMutex};
  auto& pool = g_pools[allocation.pool];
  const auto block = std::ranges::find(pool.blocks, allocation.memory, &MemoryBlock::memory);
  if (block == pool.blocks.end()) {
    Log.error("Freeing memory from no block of {}", pool.name);
    return;
  }
#ifdef AURORA_DIAGNOSTICS
  pool.used -= allocation.size;
#endif
  VkDeviceSize start = allocation.offset;
  VkDeviceSize end = allocation.offset + allocation.size;
  auto next = block->free.lower_bound(start);
  if (next != block->free.end() && next->first == end) {
    end += next->second;
    next = block->free.erase(next);
  }
  if (next != block->free.begin()) {
    const auto previous = std::prev(next);
    if (previous->first + previous->second == start) {
      start = previous->first;
      block->free.erase(previous);
    }
  }
  block->free.emplace(start, end - start);
  // An empty block goes back to the layer, but one is kept, so a load's uploads don't take and give back a kernel block
  // every frame.
  if (!block->empty() || std::ranges::count_if(pool.blocks, &MemoryBlock::empty) < 2) {
    return;
  }
  vkFreeMemory(g_device, block->memory, nullptr);
#ifdef AURORA_DIAGNOSTICS
  pool.taken -= block->size;
#endif
  pool.blocks.erase(block);
  given_back(pool);
}

bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool videoRam, const char* label,
                   Buffer& out) noexcept {
  const VkBufferCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  if (!succeeded(vkCreateBuffer(g_device, &info, nullptr, &out.buffer), "vkCreateBuffer")) {
    return false;
  }
  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(g_device, out.buffer, &requirements);
  const uint32_t poolIndex = fits_cached(requirements.memoryTypeBits) ? CachedPool : videoRam ? VideoPool : MainPool;
  const VkMemoryAllocateInfo allocateInfo{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = g_pools[poolIndex].type,
  };
  void* mapped = nullptr;
  if (!succeeded(vkAllocateMemory(g_device, &allocateInfo, nullptr, &out.memory), "vkAllocateMemory") ||
      !succeeded(vkBindBufferMemory(g_device, out.buffer, out.memory, 0), "vkBindBufferMemory") ||
      !succeeded(vkMapMemory(g_device, out.memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory")) {
    Log.error("No {} bytes of {} for {}", size, g_pools[poolIndex].name, label);
    destroy_buffer(out);
    return false;
  }
  out.mapped = static_cast<uint8_t*>(mapped);
  out.size = size;
  out.pool = poolIndex;
#ifdef AURORA_DIAGNOSTICS
  out.allocated = align_up(requirements.size, g_pools[poolIndex].granularity);
  std::lock_guard lock{g_memoryMutex};
  count_taken(g_pools[poolIndex], out.allocated);
  count_used(g_pools[poolIndex], size);
#endif
  return true;
}

void destroy_buffer(Buffer& buffer) noexcept {
  if (buffer.mapped != nullptr) {
    std::lock_guard lock{g_memoryMutex};
    auto& pool = g_pools[buffer.pool];
#ifdef AURORA_DIAGNOSTICS
    pool.taken -= buffer.allocated;
    pool.used -= buffer.size;
#endif
    given_back(pool);
  }
  vkDestroyBuffer(g_device, buffer.buffer, nullptr);
  vkFreeMemory(g_device, buffer.memory, nullptr);
  buffer = {};
}

void retire(std::function<void()> release) noexcept {
  if (g_device == VK_NULL_HANDLE) {
    return;
  }
  std::lock_guard lock{g_retiredMutex};
  g_retired.push_back({.serial = g_submitted.load(std::memory_order_acquire) + 1, .release = std::move(release)});
}

void collect_retired() noexcept { run_retired(g_completed.load(std::memory_order_acquire)); }

uint64_t next_frame_serial() noexcept { return g_submitted.load(std::memory_order_acquire) + 1; }

bool frame_done(uint64_t serial) noexcept { return g_completed.load(std::memory_order_acquire) >= serial; }

float max_anisotropy() noexcept { return g_maxAnisotropy; }

bool wait_for_frame(uint64_t serial) noexcept {
  if (serial > g_submitted.load(std::memory_order_acquire)) {
    Log.error("Not waiting for frame {}: the last submitted is {}", serial, g_submitted.load());
    return false;
  }
  // Rare, so the waiter's wake-ups stay as they are: a look every 100 us.
  while (g_completed.load(std::memory_order_acquire) < serial) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return true;
}

VkRenderPass scene_pass(Load color, Load depth, DepthStore depthStore, bool fromRead) noexcept {
  return g_scenePasses[static_cast<size_t>(depthStore)][fromRead][static_cast<size_t>(color)]
                      [static_cast<size_t>(depth)];
}

VkRenderPass color_pass(Load color) noexcept { return g_colorPasses[static_cast<size_t>(color)]; }

VkRenderPass present_pass(Load color) noexcept { return g_presentPasses[static_cast<size_t>(color)]; }

VkExtent2D swapchain_extent() noexcept { return g_swapchainExtent; }

bool samples_bc1() noexcept {
  constexpr VkFormatFeatureFlags Needed =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
  VkFormatProperties properties{};
  vkGetPhysicalDeviceFormatProperties(g_physicalDevice, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, &properties);
  return (properties.optimalTilingFeatures & Needed) == Needed;
}

bool acquire_swapchain_image(uint32_t& index, VkFramebuffer& framebuffer) noexcept {
  if (g_swapchain == VK_NULL_HANDLE) {
    return false;
  }
  if (!succeeded(vkAcquireNextImageKHR(g_device, g_swapchain, UINT64_MAX, VK_NULL_HANDLE, VK_NULL_HANDLE, &index),
                 "vkAcquireNextImageKHR")) {
    return false;
  }
  framebuffer = g_swapchainFramebuffers[index];
  return true;
}

VkCommandBuffer slot_commands(size_t slot) noexcept { return g_commands[slot]; }

bool acquire_slot(size_t slot) noexcept {
  g_slotFree[slot].wait();
  return true;
}

void release_slot(size_t slot) noexcept { g_slotFree[slot].post(); }

bool submit_frame(size_t slot, uint32_t imageIndex, bool present, [[maybe_unused]] uint64_t frameId) noexcept {
  const VkCommandBuffer commands = g_commands[slot];
  const VkFence fence = g_fences[slot];
  const bool ended = succeeded(vkEndCommandBuffer(commands), "vkEndCommandBuffer");
  succeeded(vkResetFences(g_device, 1, &fence), "vkResetFences");
  const VkSubmitInfo submit{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = ended ? 1u : 0u,
      .pCommandBuffers = &commands,
  };
#ifdef AURORA_DIAGNOSTICS
  // Taken first, so the GPU can't start the frame before the time the waiter counts from.
  const auto submittedAt = std::chrono::steady_clock::now();
  note_frame_path(frameId, FramePath::Submitted, submittedAt);
#endif
  // A fence no submission took returns VK_TIMEOUT at once (vulkan4xenon's docs/SUPPORT.md, Sync), so the waiter never
  // hangs.
  const bool submitted = succeeded(vkQueueSubmit(g_queue, 1, &submit, fence), "vkQueueSubmit");
  const uint64_t serial = g_submitted.load(std::memory_order_relaxed) + 1;
  {
    std::lock_guard lock{g_pendingMutex};
#ifdef AURORA_DIAGNOSTICS
    g_pending.push_back({.fence = fence, .serial = serial, .slot = slot, .frameId = frameId, .submitted = submittedAt});
#else
    g_pending.push_back({.fence = fence, .serial = serial, .slot = slot});
#endif
  }
  g_submitted.store(serial, std::memory_order_release);
  g_pendingCount.post();
  if (!ended || !submitted || !present) {
    return false;
  }
  const VkPresentInfoKHR info{
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .swapchainCount = 1,
      .pSwapchains = &g_swapchain,
      .pImageIndices = &imageIndex,
  };
  return succeeded(vkQueuePresentKHR(g_queue, &info), "vkQueuePresentKHR");
}
} // namespace aurora::gfx::vulkan4xenon

#ifdef AURORA_DIAGNOSTICS
// The GPU's time since the last call, frame by frame as wait_for_frames measures it.
extern "C" unsigned long long aurora_gpu_take_busy_ns() {
  return aurora::gfx::vulkan4xenon::g_gpuBusyNs.exchange(0, std::memory_order_relaxed);
}

// Memory taken from vulkan4xenon at peak by each pool, and the part of it in use at peak.
extern "C" void aurora_vulkan4xenon_memory(unsigned long long* videoTaken, unsigned long long* videoUsed,
                                           unsigned long long* mainTaken, unsigned long long* mainUsed) {
  using namespace aurora::gfx::vulkan4xenon;
  std::lock_guard lock{g_memoryMutex};
  *videoTaken = g_pools[VideoPool].takenPeak;
  *videoUsed = g_pools[VideoPool].usedPeak;
  *mainTaken = g_pools[MainPool].takenPeak;
  *mainUsed = g_pools[MainPool].usedPeak;
}
#endif
