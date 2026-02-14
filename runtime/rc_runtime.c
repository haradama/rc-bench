#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline float* extract_ptr(void* descriptor) {
    float** ptrs = (float**)descriptor;
    return ptrs[1];
}

static inline float rc_idxhash(int i, int j) {
    uint64_t x =
        (uint64_t)(i + 1) * 1315423911ULL ^ (uint64_t)(j + 1) * 2654435761ULL;
    x ^= (x >> 13);
    x *= 0x5bd1e995ULL;
    x ^= (x >> 15);
    uint32_t m = (uint32_t)(x & 0x00FFFFFFu);
    float u01 = (float)m / 16777215.0f;
    return 2.0f * u01 - 1.0f;
}

void rc_fill2d(int64_t rank, void* desc, int d0, int d1, float scale) {
    (void)rank;
    float* out = extract_ptr(desc);
    for (int i = 0; i < d0; ++i) {
        for (int j = 0; j < d1; ++j) {
            out[i * d1 + j] = rc_idxhash(i, j) * scale;
        }
    }
}

#ifdef __cplusplus
}
#endif
