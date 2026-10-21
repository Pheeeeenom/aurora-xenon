#include "thread.hpp"

#include <SDL3/SDL_thread.h>
#include <tracy/Tracy.hpp>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#elif defined(__APPLE__)
#include <pthread.h>
#endif

#ifdef __XSDK__
#include "logging.hpp"

#include <aurora/aurora.h>
#include <xsdk/thread.h>

#include <string_view>
#endif

namespace aurora::thread {
namespace {
struct Processor {
  uint16_t group = 0;
  uint32_t number = 0;
};

struct CacheDomain {
  std::vector<Processor> processors;
  uint8_t cacheLevel = 0;
};

std::mutex sDomainMutex;
std::optional<CacheDomain> sDomain;
bool sDomainConfigured = false;

#if defined(__linux__)
bool get_current_thread_affinity(cpu_set_t& set) noexcept {
#if defined(__ANDROID__)
  return sched_getaffinity(pid_t{0}, sizeof(set), &set) == 0;
#else
  return pthread_getaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#endif
}

bool set_current_thread_affinity(const cpu_set_t& set) noexcept {
#if defined(__ANDROID__)
  return sched_setaffinity(pid_t{0}, sizeof(set), &set) == 0;
#else
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#endif
}

std::optional<std::string> read_line(const std::string& path) {
  std::ifstream input{path};
  std::string value;
  if (!std::getline(input, value) || value.empty()) {
    return std::nullopt;
  }
  return value;
}

std::vector<uint32_t> parse_cpu_list(const std::string& list) {
  std::vector<uint32_t> cpus;
  size_t tokenStart = 0;
  while (tokenStart < list.size()) {
    const size_t tokenEnd = list.find(',', tokenStart);
    const size_t end = tokenEnd == std::string::npos ? list.size() : tokenEnd;
    const size_t dash = list.find('-', tokenStart);

    uint32_t first = 0;
    const char* firstBegin = list.data() + tokenStart;
    const char* firstEnd = list.data() + (dash < end ? dash : end);
    if (std::from_chars(firstBegin, firstEnd, first).ec != std::errc{}) {
      return {};
    }

    uint32_t last = first;
    if (dash < end) {
      const char* lastBegin = list.data() + dash + 1;
      const char* lastEnd = list.data() + end;
      if (std::from_chars(lastBegin, lastEnd, last).ec != std::errc{} || last < first) {
        return {};
      }
    }

    for (uint32_t cpu = first; cpu <= last; ++cpu) {
      cpus.push_back(cpu);
      if (cpu == UINT32_MAX) {
        break;
      }
    }
    tokenStart = end + 1;
  }
  return cpus;
}

std::optional<CacheDomain> find_cache_domain() {
  const int currentCpu = sched_getcpu();
  if (currentCpu < 0) {
    return std::nullopt;
  }

  CacheDomain best;
  const std::string cacheRoot = "/sys/devices/system/cpu/cpu" + std::to_string(currentCpu) + "/cache";
  for (uint32_t index = 0; index < 32; ++index) {
    const std::string indexRoot = cacheRoot + "/index" + std::to_string(index);
    const auto levelText = read_line(indexRoot + "/level");
    const auto type = read_line(indexRoot + "/type");
    const auto cpuList = read_line(indexRoot + "/shared_cpu_list");
    if (!levelText || !type || !cpuList || (*type != "Unified" && *type != "Data")) {
      continue;
    }

    uint32_t level = 0;
    if (std::from_chars(levelText->data(), levelText->data() + levelText->size(), level).ec != std::errc{} ||
        level <= best.cacheLevel) {
      continue;
    }

    const auto cpus = parse_cpu_list(*cpuList);
    if (std::find(cpus.begin(), cpus.end(), static_cast<uint32_t>(currentCpu)) == cpus.end()) {
      continue;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (!get_current_thread_affinity(allowed)) {
      continue;
    }

    CacheDomain candidate;
    candidate.cacheLevel = static_cast<uint8_t>(level);
    for (const uint32_t cpu : cpus) {
      if (cpu < CPU_SETSIZE && CPU_ISSET(static_cast<int>(cpu), &allowed)) {
        candidate.processors.push_back({.number = cpu});
      }
    }
    if (!candidate.processors.empty()) {
      best = std::move(candidate);
    }
  }

  if (best.processors.empty()) {
    return std::nullopt;
  }
  return best;
}

bool apply_cache_domain(const CacheDomain& domain) noexcept {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (const auto& processor : domain.processors) {
    if (processor.number < CPU_SETSIZE) {
      CPU_SET(static_cast<int>(processor.number), &set);
    }
  }
  return set_current_thread_affinity(set);
}
#elif defined(_WIN32)
std::optional<CacheDomain> find_cache_domain() {
  PROCESSOR_NUMBER currentProcessor{};
  GetCurrentProcessorNumberEx(&currentProcessor);

  DWORD bufferSize = 0;
  GetLogicalProcessorInformationEx(RelationCache, nullptr, &bufferSize);
  if (bufferSize == 0 || GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    return std::nullopt;
  }

  std::vector<uint8_t> buffer(bufferSize);
  auto* first = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
  if (!GetLogicalProcessorInformationEx(RelationCache, first, &bufferSize)) {
    return std::nullopt;
  }

  CacheDomain best;
  size_t offset = 0;
  while (offset < bufferSize) {
    auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
    const auto& cache = info->Cache;
    const KAFFINITY currentMask = KAFFINITY{1} << currentProcessor.Number;
    if ((cache.Type == CacheUnified || cache.Type == CacheData) && cache.Level > best.cacheLevel &&
        cache.GroupMask.Group == currentProcessor.Group && (cache.GroupMask.Mask & currentMask) != 0) {
      GROUP_AFFINITY allowed{};
      KAFFINITY mask = cache.GroupMask.Mask;
      if (GetThreadGroupAffinity(GetCurrentThread(), &allowed) && allowed.Group == cache.GroupMask.Group) {
        mask &= allowed.Mask;
      }

      CacheDomain candidate;
      candidate.cacheLevel = cache.Level;
      constexpr uint32_t bits = sizeof(KAFFINITY) * 8;
      for (uint32_t number = 0; number < bits; ++number) {
        if ((mask & (KAFFINITY{1} << number)) != 0) {
          candidate.processors.push_back({.group = cache.GroupMask.Group, .number = number});
        }
      }
      if (!candidate.processors.empty()) {
        best = std::move(candidate);
      }
    }
    if (info->Size == 0) {
      break;
    }
    offset += info->Size;
  }

  if (best.processors.empty()) {
    return std::nullopt;
  }
  return best;
}

bool apply_cache_domain(const CacheDomain& domain) noexcept {
  GROUP_AFFINITY affinity{};
  affinity.Group = domain.processors.front().group;
  for (const auto& processor : domain.processors) {
    if (processor.group == affinity.Group && processor.number < sizeof(KAFFINITY) * 8) {
      affinity.Mask |= KAFFINITY{1} << processor.number;
    }
  }
  return SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr) != 0;
}
#else
std::optional<CacheDomain> find_cache_domain() { return std::nullopt; }
bool apply_cache_domain(const CacheDomain&) noexcept { return false; }
#endif

void pin_shared_cache() noexcept {
  std::lock_guard lock{sDomainMutex};
  if (!sDomainConfigured) {
    auto domain = find_cache_domain();
    if (domain && apply_cache_domain(*domain)) {
      sDomain = std::move(domain);
    }
    sDomainConfigured = true;
    return;
  }
  if (sDomain) {
    apply_cache_domain(*sDomain);
  }
}

SDL_ThreadPriority to_sdl_priority(Priority priority) noexcept {
  switch (priority) {
  case Priority::Low:
    return SDL_THREAD_PRIORITY_LOW;
  case Priority::Normal:
    return SDL_THREAD_PRIORITY_NORMAL;
  case Priority::High:
    return SDL_THREAD_PRIORITY_HIGH;
  }
  return SDL_THREAD_PRIORITY_NORMAL;
}

void set_thread_name(const std::string& name) noexcept {
#if defined(_WIN32)
  const int length = MultiByteToWideChar(CP_UTF8, 0, name.data(), static_cast<int>(name.size()), nullptr, 0);
  if (length <= 0) {
    return;
  }
  std::wstring wideName;
  wideName.resize(static_cast<size_t>(length));
  MultiByteToWideChar(CP_UTF8, 0, name.data(), static_cast<int>(name.size()), wideName.data(), length);
  SetThreadDescription(GetCurrentThread(), wideName.c_str());
#elif defined(__APPLE__)
  const std::string truncated = name.substr(0, 63);
  pthread_setname_np(truncated.c_str());
#elif defined(__linux__)
  const std::string truncated = name.substr(0, 15);
  pthread_setname_np(pthread_self(), truncated.c_str());
#endif
}

#ifdef __XSDK__
constexpr Module Log{"aurora::thread"};

// The hardware thread each named thread runs on, as "name=thread,...". A thread not named here stays where the kernel
// starts it, on its creator's (xsdk:SCOPE.md, milestone 26). The FIFO processor, which the game waits for every frame,
// shares its core only with threads that run in bursts; the render worker shares its own with its decode helper.
std::string sHardwareThreads =
    "Main thread=0,Aurora GPU waiter=1,Aurora FIFO processor=2,Aurora tex preconv=3,Pipeline compilation thread=3,"
    "Pipeline cache writer thread=3,Aurora render worker=4,Aurora decode helper=5";

std::optional<uint32_t> hardware_thread_for(std::string_view name) noexcept {
  std::string_view list = sHardwareThreads;
  while (!list.empty()) {
    const size_t comma = list.find(',');
    const std::string_view entry = list.substr(0, comma);
    list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
    const size_t equals = entry.rfind('=');
    if (equals == std::string_view::npos || entry.substr(0, equals) != name) {
      continue;
    }
    const std::string_view value = entry.substr(equals + 1);
    uint32_t thread = 0;
    if (std::from_chars(value.data(), value.data() + value.size(), thread).ec == std::errc{}) {
      return thread;
    }
  }
  return std::nullopt;
}

void place(const std::string& name) noexcept {
  const auto thread = hardware_thread_for(name);
  if (!thread) {
    return;
  }
  if (const NTSTATUS status = xsdk_thread_move(*thread); !NT_SUCCESS(status)) {
    Log.warn("{} not moved to hardware thread {}: {:#010x}", name, *thread, static_cast<uint32_t>(status));
    return;
  }
  Log.info("{} on hardware thread {}", name, xsdk_thread_hardware_thread());
}
#endif
} // namespace

void set_current(const Options& options) noexcept {
  if (!options.name.empty()) {
    set_thread_name(options.name);
#ifdef TRACY_ENABLE
    tracy::SetThreadName(options.name.c_str());
#endif
#ifdef __XSDK__
    place(options.name);
#endif
  }

  SDL_SetCurrentThreadPriority(to_sdl_priority(options.priority));
  if (options.affinity == Affinity::SharedCache) {
    pin_shared_cache();
  }
}

} // namespace aurora::thread

#ifdef __XSDK__
void aurora_set_hardware_threads(const char* list) {
  if (list != nullptr) {
    aurora::thread::sHardwareThreads = list;
  }
}
#endif
