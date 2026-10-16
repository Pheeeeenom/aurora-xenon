#pragma once

#include "types.hpp"

// The streaming pools' sizes, the CMake settings AURORA_*_POOL_SIZE; the defaults are theirs.
#ifndef AURORA_UNIFORM_POOL_SIZE
#define AURORA_UNIFORM_POOL_SIZE 25165824 // 24 MiB
#endif
#ifndef AURORA_VERTEX_POOL_SIZE
#define AURORA_VERTEX_POOL_SIZE 5242880 // 5 MiB
#endif
#ifndef AURORA_INDEX_POOL_SIZE
#define AURORA_INDEX_POOL_SIZE 2097152 // 2 MiB
#endif
#ifndef AURORA_STORAGE_POOL_SIZE
#define AURORA_STORAGE_POOL_SIZE 8388608 // 8 MiB
#endif
#ifndef AURORA_TEXTURE_UPLOAD_POOL_SIZE
#define AURORA_TEXTURE_UPLOAD_POOL_SIZE 25165824 // 24 MiB
#endif

namespace aurora::gfx {
inline constexpr bool UseTextureBuffer = true;
inline constexpr uint64_t UniformBufferSize = AURORA_UNIFORM_POOL_SIZE;
inline constexpr uint64_t VertexBufferSize = AURORA_VERTEX_POOL_SIZE;
inline constexpr uint64_t IndexBufferSize = AURORA_INDEX_POOL_SIZE;
inline constexpr uint64_t StorageBufferSize = AURORA_STORAGE_POOL_SIZE;
inline constexpr uint64_t TextureUploadSize = AURORA_TEXTURE_UPLOAD_POOL_SIZE;
} // namespace aurora::gfx
