// sqlite's OS layer on xsdk, built with SQLITE_OS_OTHER: no VFS of its own, so databases open only through one Aurora
// registers.
#include <sqlite3.h>

int sqlite3_os_init(void) { return SQLITE_OK; }

int sqlite3_os_end(void) { return SQLITE_OK; }
