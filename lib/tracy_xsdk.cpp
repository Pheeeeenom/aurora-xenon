// Stands in for Tracy's platform layer on xsdk, while profiling is off. The kernel keeps no thread names.
#include <common/TracySystem.hpp>

namespace tracy {
void SetThreadName(const char*) {}
} // namespace tracy
