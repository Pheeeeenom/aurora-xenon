# xsdk's newlib has neither struct tm's tm_gmtoff nor <link.h>. cctz reads its _timezone and tzname as it does for
# Native Client and Emscripten, and there are no ELF images in memory to read.
function(aurora_patch_abseil file from to)
  file(READ "${file}" _content)
  string(FIND "${_content}" "${from}" _at)
  if (_at EQUAL -1)
    message(FATAL_ERROR "PatchAbseil.cmake: '${from}' not found in ${file}")
  endif ()
  string(REPLACE "${from}" "${to}" _content "${_content}")
  file(WRITE "${file}" "${_content}")
endfunction()

aurora_patch_abseil(absl/time/internal/cctz/src/time_zone_libc.cc
    "defined(__EMSCRIPTEN__)\n// Uses the globals: '_timezone' and 'tzname'."
    "defined(__EMSCRIPTEN__) || defined(__XSDK__)\n// Uses the globals: '_timezone' and 'tzname'.")
aurora_patch_abseil(absl/debugging/internal/elf_mem_image.h
    "!defined(__hexagon__)"
    "!defined(__hexagon__) && !defined(__XSDK__)")
