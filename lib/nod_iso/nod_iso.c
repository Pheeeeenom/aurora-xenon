// nod's C API over plain GameCube ISOs, for targets its Rust doesn't build for. The layout is YAGCD's
// (https://www.gc-forever.com/yagcd/chap13.html): boot.bin at 0, bi2.bin after it, the apploader at 0x2440, and
// main.dol and the FST where boot.bin points. Each call answers as nod's does (encounter/nod, nod-ffi/src/lib.rs).
#include <nod.h>

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define BOOT_SIZE 0x440
#define BI2_SIZE 0x2000
#define APPLOADER_OFFSET (BOOT_SIZE + BI2_SIZE)
#define APPLOADER_HEADER_SIZE 0x20
#define DOL_HEADER_SIZE 0x100
#define DOL_SECTIONS 18
#define FST_NODE_SIZE 12

static const char* s_errorMessage = NULL;

// The image, shared by the disc and every handle opened from it, and closed with the last of them.
struct Source {
  struct NodDiscStream stream;
  // read_at seeks and reads one stream, and Aurora reads from its DVD thread and from the thread that waits on it.
  pthread_mutex_t lock;
  int refs;
  uint64_t size;
};

enum HandleKind { HANDLE_DISC, HANDLE_PARTITION, HANDLE_FILE };

struct NodHandle {
  enum HandleKind kind;
  struct Source* source;
  // Where the handle's bytes lie on the disc; a GameCube partition is the whole disc.
  uint64_t start;
  uint64_t length;
  uint64_t position;
  struct NodDiscHeader header;  // A disc's
  struct NodPartitionMeta meta; // A partition's, each blob malloc'd
  uint32_t node_count;          // A partition's
};

static uint32_t read_be32(const uint8_t* bytes) {
  return (uint32_t)bytes[0] << 24 | (uint32_t)bytes[1] << 16 | (uint32_t)bytes[2] << 8 | bytes[3];
}

static int64_t read_at(struct Source* source, uint64_t offset, void* out, size_t len) {
  pthread_mutex_lock(&source->lock);
  const int64_t read = source->stream.read_at(source->stream.user_data, offset, out, len);
  pthread_mutex_unlock(&source->lock);
  return read;
}

static bool read_exact(struct Source* source, uint64_t offset, void* out, size_t len) {
  size_t total = 0;
  while (total < len) {
    const int64_t read = read_at(source, offset + total, (uint8_t*)out + total, len - total);
    if (read <= 0) {
      return false;
    }
    total += (size_t)read;
  }
  return true;
}

static void free_source(struct Source* source) {
  source->stream.close(source->stream.user_data);
  pthread_mutex_destroy(&source->lock);
  free(source);
}

static struct NodHandle* new_handle(enum HandleKind kind, struct Source* source, uint64_t start, uint64_t length) {
  struct NodHandle* handle = calloc(1, sizeof(*handle));
  if (handle == NULL) {
    return NULL;
  }
  pthread_mutex_lock(&source->lock);
  source->refs++;
  pthread_mutex_unlock(&source->lock);
  handle->kind = kind;
  handle->source = source;
  handle->start = start;
  handle->length = length;
  return handle;
}

static enum NodResult fail(enum NodResult result, const char* message) {
  s_errorMessage = message;
  return result;
}

const char* nod_error_message(void) { return s_errorMessage; }

enum NodResult nod_disc_open_stream(const struct NodDiscStream* stream, const struct NodDiscOptions* options,
                                    struct NodHandle** out) {
  (void)options;
  if (out == NULL) {
    return fail(NOD_RESULT_ERR_INVALID_HANDLE, "nod_iso: no handle to write");
  }
  *out = NULL;
  if (stream == NULL || stream->read_at == NULL || stream->stream_len == NULL || stream->close == NULL) {
    return fail(NOD_RESULT_ERR_INVALID_HANDLE, "nod_iso: a stream callback is missing");
  }

  // From here the stream is ours, so a failure closes it, as dropping nod's reader does.
  struct Source* source = calloc(1, sizeof(*source));
  if (source == NULL) {
    stream->close(stream->user_data);
    return fail(NOD_RESULT_ERR_OTHER, "nod_iso: out of memory");
  }
  source->stream = *stream;
  pthread_mutex_init(&source->lock, NULL);

  const int64_t size = stream->stream_len(stream->user_data);
  struct NodDiscHeader header;
  enum NodResult result = NOD_RESULT_OK;
  if (size < 0) {
    result = fail(NOD_RESULT_ERR_IO, "nod_iso: the image has no length");
  } else if (!read_exact(source, 0, &header, sizeof(header))) {
    result = fail(NOD_RESULT_ERR_IO, "nod_iso: the disc header can't be read");
  } else if (NOD_MAGIC_EQ(header.wii_magic, WII_MAGIC)) {
    result = fail(NOD_RESULT_ERR_FORMAT, "nod_iso: Wii discs aren't read here, only GameCube ones");
  } else if (!NOD_MAGIC_EQ(header.gcn_magic, GCN_MAGIC)) {
    result = fail(NOD_RESULT_ERR_FORMAT, "nod_iso: not a plain GameCube disc image");
  }
  source->size = (uint64_t)size;

  struct NodHandle* disc = result == NOD_RESULT_OK ? new_handle(HANDLE_DISC, source, 0, source->size) : NULL;
  if (disc == NULL) {
    free_source(source);
    return result != NOD_RESULT_OK ? result : fail(NOD_RESULT_ERR_OTHER, "nod_iso: out of memory");
  }
  disc->header = header;
  *out = disc;
  return NOD_RESULT_OK;
}

// Reads a blob the partition keeps, checking that it lies on the disc.
static bool read_blob(struct NodHandle* partition, uint64_t offset, uint64_t size, struct NodBlob* out) {
  if (size == 0 || offset > partition->source->size || size > partition->source->size - offset) {
    return false;
  }
  uint8_t* data = malloc((size_t)size);
  if (data == NULL || !read_exact(partition->source, offset, data, (size_t)size)) {
    free(data);
    return false;
  }
  out->data = data;
  out->size = (size_t)size;
  return true;
}

static enum NodResult read_meta(struct NodHandle* partition) {
  struct NodPartitionMeta* meta = &partition->meta;
  if (!read_blob(partition, 0, BOOT_SIZE, &meta->raw_boot) ||
      !read_blob(partition, BOOT_SIZE, BI2_SIZE, &meta->raw_bi2)) {
    return fail(NOD_RESULT_ERR_IO, "nod_iso: boot.bin or bi2.bin can't be read");
  }

  // The apploader's header gives its size at 0x14 and its trailer's at 0x18.
  uint8_t apploader[APPLOADER_HEADER_SIZE];
  if (!read_exact(partition->source, APPLOADER_OFFSET, apploader, sizeof(apploader)) ||
      !read_blob(partition, APPLOADER_OFFSET,
                 (uint64_t)APPLOADER_HEADER_SIZE + read_be32(apploader + 0x14) + read_be32(apploader + 0x18),
                 &meta->raw_apploader)) {
    return fail(NOD_RESULT_ERR_IO, "nod_iso: the apploader can't be read");
  }

  // main.dol ends where its furthest section does: 18 offsets at 0, 18 sizes at 0x90.
  const uint32_t dol_offset = read_be32(meta->raw_boot.data + 0x420);
  uint8_t dol[DOL_HEADER_SIZE];
  if (!read_exact(partition->source, dol_offset, dol, sizeof(dol))) {
    return fail(NOD_RESULT_ERR_IO, "nod_iso: main.dol's header can't be read");
  }
  uint64_t dol_size = 0;
  for (int i = 0; i < DOL_SECTIONS; i++) {
    const uint64_t end = (uint64_t)read_be32(dol + i * 4) + read_be32(dol + 0x90 + i * 4);
    if (end > dol_size) {
      dol_size = end;
    }
  }
  if (!read_blob(partition, dol_offset, dol_size > 0 ? dol_size : DOL_HEADER_SIZE, &meta->raw_dol)) {
    return fail(NOD_RESULT_ERR_IO, "nod_iso: main.dol can't be read");
  }

  if (!read_blob(partition, read_be32(meta->raw_boot.data + 0x424), read_be32(meta->raw_boot.data + 0x428),
                 &meta->raw_fst)) {
    return fail(NOD_RESULT_ERR_IO, "nod_iso: the FST can't be read");
  }
  // The root is a directory whose length is the number of nodes; the names follow the nodes.
  const uint8_t* root = meta->raw_fst.data;
  const uint32_t node_count = read_be32(root + 8);
  if (meta->raw_fst.size < FST_NODE_SIZE || root[0] != 1 || node_count == 0 ||
      node_count > meta->raw_fst.size / FST_NODE_SIZE) {
    return fail(NOD_RESULT_ERR_FORMAT, "nod_iso: the FST's root is malformed");
  }
  partition->node_count = node_count;
  return NOD_RESULT_OK;
}

enum NodResult nod_disc_open_partition_kind(struct NodHandle* disc, uint32_t kind,
                                            const struct NodPartitionOptions* options, struct NodHandle** out) {
  (void)options;
  if (out == NULL || disc == NULL || disc->kind != HANDLE_DISC) {
    return fail(NOD_RESULT_ERR_INVALID_HANDLE, "nod_iso: not a disc handle");
  }
  *out = NULL;
  if (kind != NOD_PARTITION_KIND_DATA) {
    return fail(NOD_RESULT_ERR_NOT_FOUND, "nod_iso: a GameCube disc has only its data partition");
  }

  struct NodHandle* partition = new_handle(HANDLE_PARTITION, disc->source, 0, disc->source->size);
  if (partition == NULL) {
    return fail(NOD_RESULT_ERR_OTHER, "nod_iso: out of memory");
  }
  const enum NodResult result = read_meta(partition);
  if (result != NOD_RESULT_OK) {
    nod_free(partition);
    return result;
  }
  *out = partition;
  return NOD_RESULT_OK;
}

enum NodResult nod_partition_open_file(struct NodHandle* partition, uint32_t fst_index, struct NodHandle** out) {
  if (out == NULL || partition == NULL || partition->kind != HANDLE_PARTITION) {
    return fail(NOD_RESULT_ERR_INVALID_HANDLE, "nod_iso: not a partition handle");
  }
  *out = NULL;
  if (fst_index >= partition->node_count) {
    return fail(NOD_RESULT_ERR_NOT_FOUND, "nod_iso: no such FST node");
  }
  const uint8_t* node = partition->meta.raw_fst.data + (size_t)fst_index * FST_NODE_SIZE;
  if (node[0] != 0) {
    return fail(NOD_RESULT_ERR_FORMAT, "node is not a file");
  }
  // A file's node holds its offset on the disc and its length.
  struct NodHandle* file = new_handle(HANDLE_FILE, partition->source, read_be32(node + 4), read_be32(node + 8));
  if (file == NULL) {
    return fail(NOD_RESULT_ERR_OTHER, "nod_iso: out of memory");
  }
  *out = file;
  return NOD_RESULT_OK;
}

void nod_free(struct NodHandle* handle) {
  if (handle == NULL) {
    return;
  }
  if (handle->kind == HANDLE_PARTITION) {
    free((void*)handle->meta.raw_boot.data);
    free((void*)handle->meta.raw_bi2.data);
    free((void*)handle->meta.raw_apploader.data);
    free((void*)handle->meta.raw_dol.data);
    free((void*)handle->meta.raw_fst.data);
  }
  struct Source* source = handle->source;
  free(handle);
  pthread_mutex_lock(&source->lock);
  const int refs = --source->refs;
  pthread_mutex_unlock(&source->lock);
  if (refs == 0) {
    free_source(source);
  }
}

int64_t nod_read(struct NodHandle* handle, uint8_t* buf, size_t len) {
  if (handle == NULL || buf == NULL) {
    fail(NOD_RESULT_ERR_INVALID_HANDLE, "nod_iso: nothing to read from or into");
    return -1;
  }
  if (handle->position >= handle->length) {
    return 0;
  }
  const uint64_t left = handle->length - handle->position;
  const int64_t read = read_at(handle->source, handle->start + handle->position, buf, len < left ? len : (size_t)left);
  if (read < 0) {
    fail(NOD_RESULT_ERR_IO, "nod_iso: the image can't be read");
    return -1;
  }
  handle->position += (uint64_t)read;
  return read;
}

int64_t nod_seek(struct NodHandle* handle, int64_t offset, int32_t whence) {
  if (handle == NULL) {
    fail(NOD_RESULT_ERR_INVALID_HANDLE, "nod_iso: nothing to seek");
    return -1;
  }
  int64_t base;
  switch (whence) {
  case 0:
    base = 0;
    break;
  case 1:
    base = (int64_t)handle->position;
    break;
  case 2:
    base = (int64_t)handle->length;
    break;
  default:
    fail(NOD_RESULT_ERR_OTHER, "nod_iso: no such seek origin");
    return -1;
  }
  // Past the end is allowed, as for any Rust Seek, and reads there return 0.
  if (offset < -base) {
    fail(NOD_RESULT_ERR_OTHER, "nod_iso: seek before the start");
    return -1;
  }
  handle->position = (uint64_t)(base + offset);
  return (int64_t)handle->position;
}

enum NodResult nod_disc_header(const struct NodHandle* disc, struct NodDiscHeader* out) {
  if (disc == NULL || out == NULL || disc->kind != HANDLE_DISC) {
    return fail(NOD_RESULT_ERR_INVALID_HANDLE, "nod_iso: not a disc handle");
  }
  *out = disc->header;
  return NOD_RESULT_OK;
}

uint64_t nod_disc_size(const struct NodHandle* disc) {
  return disc != NULL && disc->kind == HANDLE_DISC ? disc->source->size : 0;
}

enum NodResult nod_partition_meta(const struct NodHandle* partition, struct NodPartitionMeta* out) {
  if (partition == NULL || out == NULL || partition->kind != HANDLE_PARTITION) {
    return fail(NOD_RESULT_ERR_INVALID_HANDLE, "nod_iso: not a partition handle");
  }
  *out = partition->meta;
  return NOD_RESULT_OK;
}

// A node's name, from the table after the nodes, passed as its bytes; "<invalid>", as nod gives, if it runs off the
// FST.
static const char* node_name(const struct NodHandle* partition, const uint8_t* node) {
  const char* fst = (const char*)partition->meta.raw_fst.data;
  const size_t size = partition->meta.raw_fst.size;
  const size_t offset =
      (size_t)partition->node_count * FST_NODE_SIZE + ((size_t)node[1] << 16 | (size_t)node[2] << 8 | node[3]);
  if (offset >= size || memchr(fst + offset, '\0', size - offset) == NULL) {
    return "<invalid>";
  }
  return fst + offset;
}

void nod_partition_iterate_fst(const struct NodHandle* partition, NodFstCallback callback, void* user_data) {
  if (partition == NULL || callback == NULL || partition->kind != HANDLE_PARTITION) {
    return;
  }
  // The root, node 0, isn't visited. A directory's length is the index of the node after its last child, and the
  // callback returns the next index to visit.
  uint32_t index = 1;
  while (index < partition->node_count) {
    const uint8_t* node = partition->meta.raw_fst.data + (size_t)index * FST_NODE_SIZE;
    const enum NodNodeKind kind = node[0] != 0 ? NOD_NODE_KIND_DIRECTORY : NOD_NODE_KIND_FILE;
    index = callback(index, kind, node_name(partition, node), read_be32(node + 8), user_data);
  }
}
