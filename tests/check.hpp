#pragma once
// Test assertion that survives Release (-DNDEBUG) builds. ctest runs the
// Release tree, where plain assert() compiles to nothing and a test can
// "pass" while checking nothing.
#include <cstdio>
#include <cstdlib>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__,     \
                         __LINE__, #cond);                                 \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)
