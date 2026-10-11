#pragma once

#include <gtest/gtest.h>

#include <dolphin/gx.h>

#include "gx/gx.hpp"
#include "gx/fifo.hpp"
#include "gx/command_processor.hpp"
#include "gfx/depth_peek.hpp"

#include <cstring>
#include <vector>

// Forward declare dirty state flush
extern "C" void __GXSetDirtyState();

namespace aurora::gfx {
// A frame boundary the FIFO processor reached, with what it found there, for the frame tests (gx_test_stubs.cpp).
struct TestFrameEvent {
  enum Kind : uint8_t { Begin, End, Close } kind;
  size_t slot;
  uint32_t draws;
  AuroraViewportPolicy viewportPolicy;
  Vec2<uint32_t> viSize;
  u32 bpReg41;
};
extern std::vector<TestFrameEvent> g_testFrameEvents;

namespace testing {
// The last resolve_pass_into the FIFO processor made (gx_test_stubs.cpp).
struct ResolveCall {
  bool target = false;
  ClipRect rect{};
  bool clearColor = false;
  bool clearAlpha = false;
  bool clearDepth = false;
  Vec4<float> clearColorValue{};
  float clearDepthValue = 0.f;
};
extern ResolveCall lastResolve;
} // namespace testing
} // namespace aurora::gfx

class GXFifoTest : public ::testing::Test {
protected:
  void SetUp() override {
    GXInit(nullptr, 0);
    aurora::gx::fifo::clear_buffer();
    aurora::gx::g_gxState = aurora::gx::GXState{};
    aurora::gfx::depth_peek::testing::reset();
  }

  void TearDown() override { aurora::gx::fifo::shutdown(); }

  // Copy the internal FIFO buffer contents and clear it
  std::vector<u8> capture_fifo() {
    auto size = aurora::gx::fifo::get_buffer_size();
    auto* data = aurora::gx::fifo::get_buffer_data();
    std::vector<u8> bytes(data, data + size);
    aurora::gx::fifo::clear_buffer();
    return bytes;
  }

  // Reset g_gxState to default-constructed state
  void reset_gx_state() { aurora::gx::g_gxState = aurora::gx::GXState{}; }

  // Decode a captured FIFO byte stream through the command processor
  void decode_fifo(const std::vector<u8>& bytes) {
    aurora::gx::fifo::process(bytes.data(), static_cast<u32>(bytes.size()));
  }

  // Flush dirty state, then capture the FIFO buffer
  std::vector<u8> flush_and_capture() {
    __GXSetDirtyState();
    return capture_fifo();
  }

  // Convenience reference to g_gxState
  aurora::gx::GXState& gxState() { return aurora::gx::g_gxState; }
};
