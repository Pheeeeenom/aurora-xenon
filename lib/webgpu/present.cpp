// The Dawn backend's present: the end of a frame on the render worker.
#include "backend.hpp"

#include "gpu.hpp"
#include "gpu_prof.hpp"
#include "../gfx/depth_peek.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/render_worker.hpp"
#include "../input/router.hpp"
#include "../imgui.hpp"
#include "../internal.hpp"
#include "../io.hpp"
#include "../window.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_render.h>
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <tracy/Tracy.hpp>

#define IMGUI_IMPL_WEBGPU_BACKEND_DAWN
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_sdlrenderer3.h"
#include "backends/imgui_impl_wgpu.h"

namespace aurora::webgpu {
namespace {
// "aurora", the module this code logged under before it moved here.
constexpr Module Log{"aurora"};

uint32_t clamp_scissor_coord(double value, uint32_t maximum) noexcept {
  if (!std::isfinite(value)) {
    return 0;
  }
  return static_cast<uint32_t>(std::clamp(value, 0.0, static_cast<double>(maximum)));
}

void set_present_viewport(const wgpu::RenderPassEncoder& pass, const gfx::Viewport& viewport, uint32_t surfaceWidth,
                          uint32_t surfaceHeight) noexcept {
  pass.SetViewport(viewport.left, viewport.top, viewport.width, viewport.height, viewport.znear, viewport.zfar);
  const auto scissorX = clamp_scissor_coord(std::floor(viewport.left), surfaceWidth);
  const auto scissorY = clamp_scissor_coord(std::floor(viewport.top), surfaceHeight);
  const auto scissorRight = clamp_scissor_coord(std::ceil(viewport.left + viewport.width), surfaceWidth);
  const auto scissorBottom = clamp_scissor_coord(std::ceil(viewport.top + viewport.height), surfaceHeight);
  pass.SetScissorRect(scissorX, scissorY, scissorRight - scissorX, scissorBottom - scissorY);
}

// See aurora_gpu_frame_time.
namespace gpu_time {
std::atomic<uint64_t> g_totalNs{0};
std::atomic<uint64_t> g_count{0};
std::atomic<uint64_t> g_maxNs{0};
std::atomic<uint64_t> g_lastNs{0};

uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void record(uint64_t ns) {
  g_lastNs.store(ns, std::memory_order_relaxed);
  g_totalNs.fetch_add(ns, std::memory_order_relaxed);
  g_count.fetch_add(1, std::memory_order_relaxed);
  uint64_t prev = g_maxNs.load(std::memory_order_relaxed);
  while (ns > prev && !g_maxNs.compare_exchange_weak(prev, ns, std::memory_order_relaxed)) {}
}
} // namespace gpu_time

// A presented frame read back as a binary PPM, copied on the encoder that presents it: a copy from another encoder
// trips Metal's "A command encoder is already encoding to this command buffer".
namespace capture {
struct Readback {
  wgpu::Buffer buffer;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t paddedRow = 0;
  bool bgra = false;
};

// Copies the present source into a buffer, on the frame's encoder before it is finished.
Readback record(const wgpu::CommandEncoder& encoder) {
  const auto& src = webgpu::present_source();
  if (src.texture == nullptr) {
    return {};
  }
  Readback readback{
      .width = src.size.width,
      .height = src.size.height,
      // WebGPU requires a texture copy's bytes per row to be a multiple of 256.
      .paddedRow = AURORA_ALIGN(src.size.width * 4, 256),
      .bgra = src.format == wgpu::TextureFormat::BGRA8Unorm || src.format == wgpu::TextureFormat::BGRA8UnormSrgb,
  };
  const wgpu::BufferDescriptor bufDesc{
      .label = "Frame capture readback",
      .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
      .size = static_cast<uint64_t>(readback.paddedRow) * readback.height,
  };
  readback.buffer = g_device.CreateBuffer(&bufDesc);
  if (readback.buffer == nullptr) {
    return {};
  }
  const wgpu::TexelCopyTextureInfo copySrc{
      .texture = src.texture,
      .mipLevel = 0,
      .origin = {0, 0, 0},
      .aspect = wgpu::TextureAspect::All,
  };
  const wgpu::TexelCopyBufferInfo copyDst{
      .layout = {.offset = 0, .bytesPerRow = readback.paddedRow, .rowsPerImage = readback.height},
      .buffer = readback.buffer,
  };
  const wgpu::Extent3D extent{readback.width, readback.height, 1};
  encoder.CopyTextureToBuffer(&copySrc, &copyDst, &extent);
  return readback;
}

// Maps the buffer once the frame is submitted and writes it to `path`.
void resolve(const Readback& readback, const std::string& path) {
  const uint64_t total = static_cast<uint64_t>(readback.paddedRow) * readback.height;
  bool ok = false;
  const wgpu::Future future = readback.buffer.MapAsync(
      wgpu::MapMode::Read, 0, static_cast<size_t>(total), wgpu::CallbackMode::WaitAnyOnly,
      [&ok](wgpu::MapAsyncStatus status, wgpu::StringView) { ok = status == wgpu::MapAsyncStatus::Success; });
  wgpu::FutureWaitInfo wait{};
  wait.future = future;
  webgpu::g_instance.WaitAny(1, &wait, UINT64_MAX);
  if (!ok) {
    Log.warn("Frame capture: buffer map failed");
    readback.buffer.Destroy();
    return;
  }
  const auto* mapped = static_cast<const uint8_t*>(readback.buffer.GetConstMappedRange(0, static_cast<size_t>(total)));
  const std::string header = fmt::format("P6\n{} {}\n255\n", readback.width, readback.height);
  std::vector<uint8_t> ppm(header.begin(), header.end());
  ppm.reserve(header.size() + static_cast<size_t>(readback.width) * readback.height * 3);
  for (uint32_t y = 0; mapped != nullptr && y < readback.height; ++y) {
    const uint8_t* srcRow = mapped + static_cast<size_t>(y) * readback.paddedRow;
    for (uint32_t x = 0; x < readback.width; ++x) {
      const uint8_t* px = srcRow + x * 4;
      ppm.push_back(readback.bgra ? px[2] : px[0]);
      ppm.push_back(px[1]);
      ppm.push_back(readback.bgra ? px[0] : px[2]);
    }
  }
  readback.buffer.Unmap();
  readback.buffer.Destroy();
  if (mapped != nullptr && io::write_file(io::fs_path_from_string(path), ppm)) {
    Log.info("Wrote frame capture to {}", path);
  } else {
    Log.warn("Frame capture: could not write {}", path);
  }
}
} // namespace capture
} // namespace

void present(wgpu::CommandEncoder& encoder, std::vector<AfterSubmitCallback> afterSubmitCallbacks,
             const gfx::Viewport& viewport, const imgui::DrawData& imguiDrawData, const wgpu::BindGroup& rmlBindGroup,
             bool rmlOverlay, const std::string& capturePath) {
  wgpu::Texture currentTexture;
  wgpu::TextureView currentView;
  auto surfaceStatus = wgpu::SurfaceGetCurrentTextureStatus::Error;
  {
    window::SurfaceLock surfaceLock;
    if (window::is_presentable() && g_surface) {
      ZoneScopedN("Acquire texture");
      wgpu::SurfaceTexture surfaceTexture;
      g_surface.GetCurrentTexture(&surfaceTexture);
      surfaceStatus = surfaceTexture.status;
      /* SuccessSuboptimal still hands back a usable texture -- the surface
       * merely no longer matches the compositor's preferred configuration.
       * Treating it as a failure means never presenting again on a
       * compositor that reports it persistently (Xwayland/KDE does), which
       * leaves the window frozen on a stale frame while the game runs on.
       * Present it, and ask for a reconfigure afterwards. */
      if (surfaceStatus == wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal ||
          surfaceStatus == wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
        currentTexture = std::move(surfaceTexture.texture);
        if (currentTexture) {
          currentView = currentTexture.CreateView();
        }
      }
    }
  }

  const bool canPresent = currentTexture && currentView && webgpu::g_graphicsConfig.surfaceConfiguration.width > 0 &&
                          webgpu::g_graphicsConfig.surfaceConfiguration.height > 0;
  if (canPresent) {
    wgpu::BindGroup presentBindGroup;
    if (rmlBindGroup && !rmlOverlay) {
      presentBindGroup = rmlBindGroup;
    } else if (webgpu::get_resampler() == SAMPLER_BILINEAR) {
      presentBindGroup = webgpu::g_CopyBindGroup;
    } else {
      const auto& resampledSource = webgpu::resample_present_source(encoder, viewport);
      presentBindGroup = webgpu::create_copy_bind_group(resampledSource);
    }
    {
      const std::array attachments{
          wgpu::RenderPassColorAttachment{
              .view = currentView,
              .loadOp = wgpu::LoadOp::Clear,
              .storeOp = wgpu::StoreOp::Store,
          },
      };
      const wgpu::RenderPassDescriptor renderPassDescriptor{
          .label = "EFB copy render pass",
          .colorAttachmentCount = attachments.size(),
          .colorAttachments = attachments.data(),
          .timestampWrites = webgpu::gpu_prof::pass_writes("Present blit"),
      };
      const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
      // Copy EFB -> XFB (swapchain)
      pass.SetPipeline(webgpu::g_CopyPipeline);
      pass.SetBindGroup(0, presentBindGroup, 0, nullptr);
      /* The acquired texture's own size, not the configured one: a browser
       * canvas that the page resized (SDL follows its CSS size) hands back
       * a texture of the new size before the resize event reconfigures the
       * surface, and a scissor outside it is a fatal validation error.
       * Natively the two sizes are always equal. */
      set_present_viewport(pass, viewport, currentTexture.GetWidth(), currentTexture.GetHeight());

      pass.Draw(3);
      if (rmlBindGroup && rmlOverlay) {
        pass.SetPipeline(webgpu::g_CopyPremultipliedAlphaPipeline);
        pass.SetBindGroup(0, rmlBindGroup, 0, nullptr);
        pass.Draw(3);
      }
      pass.End();
    }
    // An ImGui pass with no draw lists would still load and store the swapchain image.
    if (!imguiDrawData.empty()) {
      const std::array attachments{
          wgpu::RenderPassColorAttachment{
              .view = currentView,
              .loadOp = wgpu::LoadOp::Load,
              .storeOp = wgpu::StoreOp::Store,
          },
      };
      const wgpu::RenderPassDescriptor renderPassDescriptor{
          .label = "ImGui render pass",
          .colorAttachmentCount = attachments.size(),
          .colorAttachments = attachments.data(),
          .timestampWrites = webgpu::gpu_prof::pass_writes("ImGui"),
      };
      const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
      pass.SetViewport(0.f, 0.f, static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.width),
                       static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.height), 0.f, 1.f);
      imgui::render(pass, imguiDrawData);
      pass.End();
    }
  } else {
    Log.info("Skipping present; no usable surface texture ({})", magic_enum::enum_name(surfaceStatus));
  }
  const capture::Readback readback = capturePath.empty() ? capture::Readback{} : capture::record(encoder);
  webgpu::gpu_prof::frame_end(encoder);
  const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Redraw command buffer"};
  const auto buffer = encoder.Finish(&cmdBufDescriptor);
  {
    ZoneScopedN("Queue Submit");
    g_queue.Submit(1, &buffer);
  }
  const uint64_t submitNs = gpu_time::now_ns();
  g_queue.OnSubmittedWorkDone(wgpu::CallbackMode::AllowSpontaneous,
                              [submitNs](wgpu::QueueWorkDoneStatus status, wgpu::StringView) {
                                if (status == wgpu::QueueWorkDoneStatus::Success) {
                                  gpu_time::record(gpu_time::now_ns() - submitNs);
                                }
                              });
  if (readback.buffer) {
    capture::resolve(readback, capturePath);
  }
  webgpu::gpu_prof::after_submit();
  if (canPresent && g_surface) {
    ZoneScopedN("Present");
    wgpu::ConvertibleStatus status = wgpu::Status::Error;
    {
      window::SurfaceLock surfaceLock;
      if (window::is_presentable()) {
#ifdef __EMSCRIPTEN__
        // The browser presents the canvas when control returns to its event loop.
        status = wgpu::Status::Success;
#else
        status = g_surface.Present();
#endif
      }
    }
    if (status) {
      gfx::after_present();
      /* Only on the transition into suboptimal: refresh_surface() calls
       * gpu_synchronize(), so asking every frame would stall the pipeline
       * for as long as the compositor keeps reporting it. */
      static bool wasSuboptimal = false;
      const bool suboptimal = surfaceStatus == wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal;
      if (suboptimal && !wasSuboptimal) {
        window::push_custom_event(window::CustomEvent::RefreshSurface);
      }
      wasSuboptimal = suboptimal;
    } else {
      Log.warn("Surface present failed");
      webgpu::release_surface();
    }
  } else if (g_surface) {
    switch (surfaceStatus) {
    case wgpu::SurfaceGetCurrentTextureStatus::Timeout:
      Log.warn("Surface texture acquisition timed out");
      break;
    case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
    case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
      Log.info("Surface texture is {}, reconfiguring swapchain", magic_enum::enum_name(surfaceStatus));
      window::push_custom_event(window::CustomEvent::RefreshSurface);
      break;
    case wgpu::SurfaceGetCurrentTextureStatus::Lost:
      Log.warn("Surface texture is {}, releasing surface", magic_enum::enum_name(surfaceStatus));
      webgpu::release_surface();
      break;
    case wgpu::SurfaceGetCurrentTextureStatus::Error:
      Log.warn("Surface texture is {}, releasing surface", magic_enum::enum_name(surfaceStatus));
      webgpu::release_surface();
      break;
    default:
      if (!window::is_presentable()) {
        webgpu::release_surface();
      } else {
        Log.error("Failed to get surface texture: {}", magic_enum::enum_name(surfaceStatus));
      }
      break;
    }
  }
  for (auto& callback : afterSubmitCallbacks) {
    if (callback) {
      callback();
    }
  }
  gfx::depth_peek::after_submit();
}
} // namespace aurora::webgpu

namespace aurora::imgui {
namespace {
float g_scale;
bool g_useSdlRenderer = false;

std::vector<SDL_Texture*> g_sdlTextures;
std::vector<wgpu::Texture> g_wgpuTextures;

wgpu::Buffer create_texture_upload_buffer(uint32_t width, uint32_t height, const uint8_t* data,
                                          uint32_t copyBytesPerRow) {
  const uint32_t rowBytes = width * 4;
  const uint64_t uploadSize = static_cast<uint64_t>(copyBytesPerRow) * height;
  const wgpu::BufferDescriptor desc{
      .label = "imgui texture upload buffer",
      .usage = wgpu::BufferUsage::CopySrc,
      .size = uploadSize,
      .mappedAtCreation = true,
  };
  auto buffer = webgpu::g_device.CreateBuffer(&desc);
  auto* dst = static_cast<uint8_t*>(buffer.GetMappedRange(0, uploadSize));
  for (uint32_t row = 0; row < height; ++row) {
    std::memcpy(dst, data, rowBytes);
    dst += copyBytesPerRow;
    data += rowBytes;
  }
  return buffer;
}

void enqueue_texture_upload(wgpu::Buffer buffer, wgpu::TexelCopyTextureInfo dst, wgpu::TexelCopyBufferLayout layout,
                            wgpu::Extent3D size) {
  gfx::render_worker::enqueue_work([buffer = std::move(buffer), dst = std::move(dst), layout, size] {
    buffer.Unmap();
    const wgpu::CommandEncoderDescriptor encoderDesc{
        .label = "imgui texture upload encoder",
    };
    auto encoder = webgpu::g_device.CreateCommandEncoder(&encoderDesc);
    const wgpu::TexelCopyBufferInfo src{
        .layout = layout,
        .buffer = buffer,
    };
    encoder.CopyBufferToTexture(&src, &dst, &size);
    constexpr wgpu::CommandBufferDescriptor commandBufferDesc{
        .label = "imgui texture upload command buffer",
    };
    auto commandBuffer = encoder.Finish(&commandBufferDesc);
    webgpu::g_queue.Submit(1, &commandBuffer);
  });
}

input::LayerId g_inputLayer = input::kInvalidLayerId;

input::EventResult input_layer_event(const input::InputEvent& event, void*) {
  using Kind = input::InputSource::Kind;
  const Kind kind = event.source.kind;
  if (ImGui::GetCurrentContext() == nullptr || (kind != Kind::Keyboard && kind != Kind::Mouse)) {
    return input::EventResult::Pass;
  }
  ImGuiIO& io = ImGui::GetIO();
  if (event.payload.is<input::InputEvent::Cancelled>()) {
    if (kind == Kind::Keyboard) {
      io.ClearInputKeys();
    } else {
      io.ClearInputMouse();
    }
    return input::EventResult::Pass;
  }
  if (const SDL_Event* raw = input::detail::current_sdl_event()) {
    process_event(*raw);
  }
  const bool capture = kind == Kind::Keyboard ? io.WantCaptureKeyboard || io.WantTextInput : io.WantCaptureMouse;
  return capture ? input::EventResult::Consume : input::EventResult::Pass;
}

bool input_layer_captures(const input::InputSource& source, void*) {
  if (ImGui::GetCurrentContext() == nullptr) {
    return false;
  }
  const ImGuiIO& io = ImGui::GetIO();
  switch (source.kind) {
  case input::InputSource::Kind::Keyboard:
    return io.WantCaptureKeyboard || io.WantTextInput;
  case input::InputSource::Kind::Mouse:
    return io.WantCaptureMouse;
  default:
    return false;
  }
}
} // namespace

void initialize() noexcept {
  ZoneScoped;
  SDL_Renderer* renderer = window::get_sdl_renderer();
  ImGui_ImplSDL3_InitForSDLRenderer(window::get_sdl_window(), renderer);
  g_useSdlRenderer = renderer != nullptr;
  if (g_useSdlRenderer) {
    ImGui_ImplSDLRenderer3_Init(renderer);
  } else {
    ImGui_ImplWGPU_InitInfo info;
    info.Device = webgpu::g_device.Get();
    info.RenderTargetFormat = static_cast<WGPUTextureFormat>(webgpu::g_graphicsConfig.surfaceConfiguration.format);
    ImGui_ImplWGPU_Init(&info);
  }
  g_inputLayer = input::register_layer({
      .label = "aurora.imgui",
      .priority = input::kImGuiLayerPriority,
      .onEvent = input_layer_event,
      .capturesSource = input_layer_captures,
  });
}

void shutdown() noexcept {
  ZoneScoped;
  input::unregister_layer(g_inputLayer);
  g_inputLayer = input::kInvalidLayerId;
  if (g_useSdlRenderer) {
    ImGui_ImplSDLRenderer3_Shutdown();
  } else {
    ImGui_ImplWGPU_Shutdown();
  }
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
  for (const auto& texture : g_sdlTextures) {
    SDL_DestroyTexture(texture);
  }
  g_sdlTextures.clear();
  g_wgpuTextures.clear();
}

void process_event(const SDL_Event& event) noexcept {
  auto renderEvent = event;
  if (g_useSdlRenderer) {
    if (SDL_Renderer* renderer = window::get_sdl_renderer()) {
      SDL_ConvertEventToRenderCoordinates(renderer, &renderEvent);
    }
  }
  ImGui_ImplSDL3_ProcessEvent(&renderEvent);
}

void new_frame(const AuroraWindowSize& size) noexcept {
  ZoneScoped;
  ImVec2 framebufferScale{
      size.width > 0 ? static_cast<float>(size.native_fb_width) / static_cast<float>(size.width) : 1.0f,
      size.height > 0 ? static_cast<float>(size.native_fb_height) / static_cast<float>(size.height) : 1.0f,
  };
  ImVec2 displaySize{static_cast<float>(size.width), static_cast<float>(size.height)};

  if (g_useSdlRenderer) {
    if (SDL_Renderer* renderer = window::get_sdl_renderer()) {
      float renderScaleX = 1.0f;
      float renderScaleY = 1.0f;
      SDL_GetRenderScale(renderer, &renderScaleX, &renderScaleY);
      if (renderScaleX > 0.0f && renderScaleY > 0.0f &&
          (std::fabs(renderScaleX - 1.0f) > 0.0001f || std::fabs(renderScaleY - 1.0f) > 0.0001f)) {
        int outputWidth = static_cast<int>(size.native_fb_width);
        int outputHeight = static_cast<int>(size.native_fb_height);
        SDL_GetRenderOutputSize(renderer, &outputWidth, &outputHeight);
        displaySize = {
            static_cast<float>(outputWidth) / renderScaleX,
            static_cast<float>(outputHeight) / renderScaleY,
        };
        framebufferScale = {renderScaleX, renderScaleY};
      }
    }
    ImGui_ImplSDLRenderer3_NewFrame();
    g_scale = size.scale;
  } else {
    if (g_scale != size.scale) {
      if (g_scale > 0.f) {
        ImGui_ImplWGPU_CreateDeviceObjects();
      }
      g_scale = size.scale;
    }
    if (!ImGui::GetIO().Fonts->IsBuilt()) {
      ImGui_ImplWGPU_CreateDeviceObjects();
    }
    ImGui_ImplWGPU_NewFrame();
  }
  ImGui_ImplSDL3_NewFrame();

  ImGuiIO& io = ImGui::GetIO();
  io.DisplayFramebufferScale = framebufferScale;
  ImGui::GetIO().DisplaySize = displaySize;
  ImGui::NewFrame();
}

void render(const wgpu::RenderPassEncoder& pass, const DrawData& drawData) noexcept {
  ZoneScoped;

  auto* data = drawData.data();
  if (data == nullptr || data->CmdListsCount == 0) {
    return;
  }
  if (g_useSdlRenderer) {
    SDL_Renderer* renderer = window::get_sdl_renderer();
    SDL_RenderClear(renderer);
    ImGui_ImplSDLRenderer3_RenderDrawData(data, renderer);
    SDL_RenderPresent(renderer);
  } else {
    pass.PushDebugGroup("Aurora: Dear Imgui");
    ImGui_ImplWGPU_RenderDrawData(data, pass.Get());
    pass.PopDebugGroup();
  }
}

ImTextureID add_texture(uint32_t width, uint32_t height, const uint8_t* data) noexcept {
  if (SDL_Renderer* renderer = window::get_sdl_renderer()) {
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STATIC, width, height);
    SDL_UpdateTexture(texture, nullptr, data, width * 4);
    SDL_SetTextureScaleMode(texture, SDL_SCALEMODE_LINEAR);
    g_sdlTextures.push_back(texture);
    return reinterpret_cast<ImTextureID>(texture);
  }
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
  const wgpu::TextureDescriptor textureDescriptor{
      .label = "imgui texture",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  const wgpu::TextureViewDescriptor textureViewDescriptor{
      .label = "imgui texture view",
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = WGPU_MIP_LEVEL_COUNT_UNDEFINED,
      .arrayLayerCount = WGPU_ARRAY_LAYER_COUNT_UNDEFINED,
  };
  auto texture = webgpu::g_device.CreateTexture(&textureDescriptor);
  auto textureView = texture.CreateView(&textureViewDescriptor);
  {
    const wgpu::TexelCopyTextureInfo dstView{
        .texture = texture,
    };
    const uint32_t copyBytesPerRow = AURORA_ALIGN(width * 4, 256);
    const wgpu::TexelCopyBufferLayout dataLayout{
        .bytesPerRow = copyBytesPerRow,
        .rowsPerImage = height,
    };
    enqueue_texture_upload(create_texture_upload_buffer(width, height, data, copyBytesPerRow), dstView, dataLayout,
                           size);
  }
  g_wgpuTextures.push_back(texture);
  return reinterpret_cast<ImTextureID>(textureView.MoveToCHandle());
}
} // namespace aurora::imgui

// C bindings
extern "C" {
ImTextureID aurora_imgui_add_texture(uint32_t width, uint32_t height, const void* rgba8) {
  return aurora::imgui::add_texture(width, height, static_cast<const uint8_t*>(rgba8));
}
}

void aurora_gpu_frame_time(uint64_t* lastNs, uint64_t* meanNs, uint64_t* maxNs, uint64_t* count) {
  const uint64_t n = aurora::webgpu::gpu_time::g_count.load(std::memory_order_relaxed);
  if (lastNs != nullptr) {
    *lastNs = aurora::webgpu::gpu_time::g_lastNs.load(std::memory_order_relaxed);
  }
  if (maxNs != nullptr) {
    *maxNs = aurora::webgpu::gpu_time::g_maxNs.load(std::memory_order_relaxed);
  }
  if (count != nullptr) {
    *count = n;
  }
  if (meanNs != nullptr) {
    *meanNs = n != 0 ? aurora::webgpu::gpu_time::g_totalNs.load(std::memory_order_relaxed) / n : 0;
  }
}
