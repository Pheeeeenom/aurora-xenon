#pragma once
// The vulkan4xenon backend's vertex stream for a GX shader config: GX vertices decoded on the CPU into the layout
// glsl.hpp's programs read, and the caches that keep the decoded vertices from frame to frame.

#include "../gx/pipeline.hpp"
#include "glsl.hpp"

#include <absl/container/flat_hash_map.h>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace aurora::gfx::vulkan4xenon::decode {
using AttrFormat = glsl::AttrFormat;
using NbtSlice = glsl::NbtSlice;
using Attribute = glsl::Attribute;
using VertexLayout = glsl::VertexLayout;

// Render worker, per draw: the draw's vertices in layout's stream, from the frame's vertex and storage pools.
void decode_vertices(const gx::ShaderConfig& config, const VertexLayout& layout, const gx::DrawData& draw,
                     std::span<const uint8_t> verts, std::span<const uint8_t> storage, uint8_t* out) noexcept;

// Where each indexed attribute's array ends in the storage pool, by attribute from GX_VA_POS (gfx::push_array's sizes).
using ArrayEnds = std::array<uint32_t, gx::MaxIndexAttr>;
// None known: reads stop only at the end of the storage pool.
inline constexpr ArrayEnds UnknownArrayEnds = [] {
  ArrayEnds ends;
  ends.fill(UINT32_MAX);
  return ends;
}();

// The same vertices, and whether every array element the decode read lay wholly inside its attribute's array. An
// index past its array reads whatever the pool holds after it, which a key over the arrays alone doesn't cover.
bool decode_vertices(const gx::ShaderConfig& config, const VertexLayout& layout, const gx::DrawData& draw,
                     std::span<const uint8_t> verts, std::span<const uint8_t> storage, const ArrayEnds& ends,
                     uint8_t* out) noexcept;

// A draw's index list, from its vertex bytes alone: each vertex's place among the draw's distinct vertices (the first
// vertex with some bytes takes the next place, its repeats the same one), and the first vertex of each place: a draw's
// vertices often repeat earlier ones, and a repeat decodes for nearly what a new vertex does.
struct IndexList {
  std::vector<uint16_t> places; // a vertex's index into the packed vertices
  std::vector<uint16_t> firsts; // a place's vertex
};

// The list of count vertices of stride bytes each, up to 65,536 of them.
IndexList build_index_list(std::span<const uint8_t> vertices, uint32_t stride, uint32_t count);

// A listed draw's vertex cache entry: its places as 16-bit indices, padded to the cache's 16 bytes so the vertices
// after them start aligned, then its first vertices, packed.
constexpr uint64_t listed_indices_size(uint32_t count) noexcept { return (uint64_t{count} * 2 + 15) & ~uint64_t{15}; }
inline uint64_t listed_entry_size(const IndexList& list, uint32_t stride) noexcept {
  return listed_indices_size(static_cast<uint32_t>(list.places.size())) + uint64_t{list.firsts.size()} * stride;
}

// The draw's first vertices only, each as decode_vertices decodes it, packed: decoded vertex k is GX vertex firsts[k].
// Says whether every read stayed inside its array, as decode_vertices does.
bool decode_firsts(const gx::ShaderConfig& config, const VertexLayout& layout, const gx::DrawData& draw,
                   std::span<const uint8_t> verts, std::span<const uint8_t> storage, const ArrayEnds& ends,
                   std::span<const uint16_t> firsts, uint8_t* out) noexcept;

// One draw of a pass for decode_pass: decode_vertices' arguments but the pools, which the pass's draws share, and
// whether its reads stayed inside its arrays. With a list, out takes the list's places, then its first vertices.
struct DecodeJob {
  const gx::ShaderConfig* config;
  const VertexLayout* layout;
  const gx::DrawData* draw;
  uint8_t* out;
  ArrayEnds ends = UnknownArrayEnds;
  bool inside = true;
  const IndexList* list = nullptr;
};

// Render worker, per pass: every job's vertices as decode_vertices writes them, decoded by the caller and a helper
// thread on its own core, which take the draws in turn. It returns once all are written.
void decode_pass(std::span<DecodeJob> jobs, std::span<const uint8_t> verts, std::span<const uint8_t> storage) noexcept;

// Render worker: since the last call, the passes decode_pass shared with its helper, their vertices, and the helper's
// share of them.
struct HelperStats {
  uint64_t passes = 0;
  uint64_t vertices = 0;
  uint64_t helperVertices = 0;
};
HelperStats take_helper_stats() noexcept;

// The vertex cache's key for a draw: a hash of everything decode_vertices reads for it, so that draws with equal keys
// decode to equal vertices.
struct DecodeKey {
  uint64_t low;
  uint64_t high;
  bool operator==(const DecodeKey&) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const DecodeKey& key) {
    return H::combine(std::move(h), key.low);
  }
};

// Render worker: index lists by the vertex bytes they're built from (their hash, count and stride), so a list is built
// when a draw first meets those bytes, not each time it misses the vertex cache. Only the decode reads them, as a
// listed entry holds its own copy of the places for the GPU, so a list may go whenever no pass is being laid out.
class IndexLists {
public:
  explicit IndexLists(uint64_t capacity) noexcept : m_capacity(capacity) {}

  // The list for a draw's vertex bytes, count vertices of stride bytes whose XXH3-64 is hash, built if none is kept.
  // It stays valid until the next begin_frame.
  const IndexList& get(uint64_t hash, std::span<const uint8_t> vertices, uint32_t stride, uint32_t count);

  // At a frame's start: past the capacity, the lists used longest ago go.
  void begin_frame() noexcept;

  uint64_t bytes() const noexcept { return m_bytes; }
  size_t size() const noexcept { return m_lists.size(); }
  // Lists built since the last call.
  uint64_t take_built() noexcept { return std::exchange(m_built, 0); }

private:
  struct Kept {
    IndexList list;
    uint64_t lastFrame;
  };
  static uint64_t kept_bytes(const IndexList& list) noexcept;

  // Held by pointer, so a list stays where it is while later draws' lists are added.
  absl::flat_hash_map<DecodeKey, std::unique_ptr<Kept>> m_lists;
  uint64_t m_capacity;
  uint64_t m_bytes = 0;
  uint64_t m_frame = 0;
  uint64_t m_built = 0;
};

// Once per pipeline: what decode_vertices reads of its config and layout, hashed: the vertex stride, and each
// attribute's GX layout and place in the decoded vertex.
uint64_t decode_format(const gx::ShaderConfig& config, const VertexLayout& layout) noexcept;

// A range of a frame's pools for hash_ranges, and its XXH3-64 once hash_ranges returns.
struct HashRange {
  const uint8_t* data;
  size_t size;
  uint64_t hash;
};

// Render worker: hashes every range, sharing them with the decode's helper thread when there are enough bytes.
void hash_ranges(std::span<HashRange> ranges) noexcept;

// One draw for VertexKeys::compute.
struct KeyedDraw {
  uint64_t format; // its pipeline's decode_format
  const gx::ShaderConfig* config;
  const VertexLayout* layout;
  const gx::DrawData* draw;
  bool listed = false; // its entry holds an index list and its first vertices
};

// Render worker: the vertex cache's keys, by content alone, so no game has to say when it changes its vertex data.
// A draw's key is its format, its vertex count, the hash of its vertex bytes as the frame's vertex pool holds them
// (for a draw from a display list, that list's bytes: KISS, and as often equal across frames as a key by the list and
// the draw's place in it), and the hash of each array it reads, whole, as push_array put it in the storage pool,
// each array hashed once a frame. A draw whose vertex bytes leave the pool gets no key, and is decoded every frame.
class VertexKeys {
public:
  // Render worker, at a frame's start. erase_if keeps the table, where clear() frees one of 128 slots or more (absl's
  // raw_hash_set.h, clear), so no frame allocates it again.
  void begin_frame() noexcept {
    absl::erase_if(m_arrayPlaces, [](const auto&) { return true; });
    m_arrays.clear();
  }

  // Each draw's key, or none, and where each of its arrays ends, for decode_vertices to say whether it read past one.
  // With poolHashes, the FIFO worker hashed the pools' bytes as it copied them (gfx::backend::Limits::contentHashes):
  // a draw's vertex bytes when its gx::DrawData::vertHash isn't zero, and every array. Those hashes are taken, and
  // only the rest are read.
  void compute(std::span<const KeyedDraw> draws, std::span<const uint8_t> verts, std::span<const uint8_t> storage,
               std::span<std::optional<DecodeKey>> keys, std::span<ArrayEnds> ends, bool poolHashes = false);

  // From the last compute: the XXH3-64 of a draw's vertex bytes as its key has it, for a draw with a key.
  uint64_t vertex_hash(size_t draw) const noexcept { return m_vertexHashes[draw]; }

private:
  struct Array {
    uint32_t end;
    uint32_t range; // its place in m_ranges while this pass hashes it, or Hashed
    uint64_t hash;
  };
  static constexpr uint32_t Hashed = UINT32_MAX;
  // The frame's arrays, and their places in m_arrays by start in the frame's storage pool. Each draw keeps its
  // attributes' places (m_drawArrays, from m_drawArraysStart), so its key doesn't look them up again.
  std::vector<Array> m_arrays;
  absl::flat_hash_map<uint32_t, uint32_t> m_arrayPlaces;
  std::vector<uint32_t> m_drawArrays;
  std::vector<uint32_t> m_drawArraysStart;
  std::vector<HashRange> m_ranges;
  std::vector<uint32_t> m_vertexRanges; // each draw's place in m_ranges, NoRange, or PoolHash
  std::vector<uint64_t> m_vertexHashes;
  std::vector<uint8_t> m_keyBytes;
  static constexpr uint32_t NoRange = UINT32_MAX;
  static constexpr uint32_t PoolHash = UINT32_MAX - 1; // the draw's vertHash
};

// Decoded vertices kept from frame to frame in one block of video RAM, found by key. Space for new ones comes from the
// least recently used, never from one a frame still in flight may draw: an entry's space is free again once the last
// frame that used it is done on the GPU. A frame is done framesInFlight frames later, as a frame slot's fence has then
// signalled, or sooner when the cache has waited for it.
class VertexCache {
public:
  // Blocks until the frame numbered so is done on the GPU, and says whether it is. The frames before the one being laid
  // out are submitted, so the wait ends.
  using FrameWaiter = std::function<bool(uint64_t frame)>;

  // Without a waiter, a draw that finds no room gets none.
  VertexCache(uint64_t capacity, uint64_t framesInFlight, FrameWaiter waitForFrame = {});

  // Render worker, at a frame's start: the frame's number, higher than the one before; without one, the next number.
  void begin_frame(uint64_t frame) noexcept;
  void begin_frame() noexcept { begin_frame(m_frame + 1); }

  // Where a draw's vertices are: the key's entry, now in use by this frame, or new space for size bytes, which the
  // caller decodes into. A draw with no key gets space that no key finds. When no space is free, it waits for the
  // oldest frame in flight to be done, frees what that frame held, and tries again, until the draw fits. None only when
  // it can't wait, or when every frame before this one is done and the draw still doesn't fit beside this frame's own:
  // a cache smaller than one frame's vertices.
  struct Place {
    uint64_t offset;
    bool decoded; // it holds the draw's vertices already
  };
  std::optional<Place> place(const std::optional<DecodeKey>& key, uint64_t size);

  // The key's entry, now in use by this frame, or none: for a draw whose size is known only once it misses.
  std::optional<Place> find(const DecodeKey& key) noexcept;

  // A key whose decode read outside what the key covers: its entry stays until its space is taken, but no draw finds
  // it again.
  void forget(const DecodeKey& key) noexcept;

  // Since the last call. A frame's bytes are those of every entry its draws used.
  struct Stats {
    uint64_t hits = 0;
    uint64_t hitBytes = 0;
    uint64_t misses = 0;
    uint64_t missBytes = 0;
    uint64_t evictions = 0;
    uint64_t failures = 0;
    uint64_t forgotten = 0;
    uint64_t peakFrameBytes = 0;
    uint64_t peakUsedBytes = 0;
    // Waits for a frame in flight to be done, and their time.
    uint64_t waits = 0;
    uint64_t waitNs = 0;
    uint64_t longestWaitNs = 0;
  };
  Stats take_stats() noexcept { return std::exchange(m_stats, Stats{}); }
  uint64_t frame_bytes() const noexcept { return m_frameBytes; }

private:
  static constexpr uint32_t None = UINT32_MAX;
  struct Entry {
    DecodeKey key;
    bool keyed;
    uint64_t offset;
    uint64_t size;
    uint64_t lastFrame;
    uint32_t newer; // in the list of entries by last use
    uint32_t older;
  };

  Place use(uint32_t index, uint64_t size) noexcept;
  void touch(uint32_t index) noexcept;
  void link_newest(uint32_t index) noexcept;
  void unlink(uint32_t index) noexcept;
  std::optional<uint64_t> allocate(uint64_t size);
  void release(uint64_t offset, uint64_t size);
  bool evict_oldest();
  bool wait_for_oldest_frame();

  uint64_t m_framesInFlight;
  FrameWaiter m_waitForFrame;
  uint64_t m_frame = 0;
  // The newest frame known done: entries it or earlier frames used last may be evicted.
  uint64_t m_done = 0;
  uint64_t m_used = 0;
  uint64_t m_frameBytes = 0;
  std::vector<Entry> m_entries;
  std::vector<uint32_t> m_freeEntries;
  uint32_t m_newest = None;
  uint32_t m_oldest = None;
  absl::flat_hash_map<DecodeKey, uint32_t> m_byKey;
  // Free space, by offset and by size, best fit.
  std::map<uint64_t, uint64_t> m_freeByOffset;
  std::multimap<uint64_t, uint64_t> m_freeBySize;
  Stats m_stats;
};
} // namespace aurora::gfx::vulkan4xenon::decode
