// nod's C API, for targets its Rust doesn't build for. No disc reader yet, so every open fails.
#include <nod.h>

static const char* s_errorMessage = NULL;

const char* nod_error_message(void) { return s_errorMessage; }

enum NodResult nod_disc_open_stream(const struct NodDiscStream* stream, const struct NodDiscOptions* options,
                                    struct NodHandle** out) {
  (void)stream;
  (void)options;
  *out = NULL;
  s_errorMessage = "nod_iso: no disc reader yet";
  return NOD_RESULT_ERR_FORMAT;
}

// No open succeeds, so no handle reaches the rest.
enum NodResult nod_disc_open_partition_kind(struct NodHandle* disc, uint32_t kind,
                                            const struct NodPartitionOptions* options, struct NodHandle** out) {
  (void)disc;
  (void)kind;
  (void)options;
  *out = NULL;
  return NOD_RESULT_ERR_INVALID_HANDLE;
}

enum NodResult nod_partition_open_file(struct NodHandle* partition, uint32_t fst_index, struct NodHandle** out) {
  (void)partition;
  (void)fst_index;
  *out = NULL;
  return NOD_RESULT_ERR_INVALID_HANDLE;
}

void nod_free(struct NodHandle* handle) { (void)handle; }

int64_t nod_read(struct NodHandle* handle, uint8_t* buf, size_t len) {
  (void)handle;
  (void)buf;
  (void)len;
  return -1;
}

int64_t nod_seek(struct NodHandle* handle, int64_t offset, int32_t whence) {
  (void)handle;
  (void)offset;
  (void)whence;
  return -1;
}

enum NodResult nod_disc_header(const struct NodHandle* disc, struct NodDiscHeader* out) {
  (void)disc;
  (void)out;
  return NOD_RESULT_ERR_INVALID_HANDLE;
}

uint64_t nod_disc_size(const struct NodHandle* disc) {
  (void)disc;
  return 0;
}

enum NodResult nod_partition_meta(const struct NodHandle* partition, struct NodPartitionMeta* out) {
  (void)partition;
  (void)out;
  return NOD_RESULT_ERR_INVALID_HANDLE;
}

void nod_partition_iterate_fst(const struct NodHandle* partition, NodFstCallback callback, void* user_data) {
  (void)partition;
  (void)callback;
  (void)user_data;
}
