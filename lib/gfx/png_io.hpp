#pragma once

#include <filesystem>
#include <optional>

#include "replacement_texture.hpp"

namespace aurora::gfx::png {
std::optional<ReplacementTexture> parse_png_bytes(ArrayRef<uint8_t> bytes) noexcept;
std::optional<ReplacementTexture> load_png_file(const std::filesystem::path& path) noexcept;
// For texture dumps.
bool write_rgba8_png(const std::filesystem::path& path, uint32_t width, uint32_t height,
                     ArrayRef<uint8_t> pixels) noexcept;
} // namespace aurora::gfx::png
