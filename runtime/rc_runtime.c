#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Helper to extract Aligned Pointer from Unranked MemRef Descriptor ---
// Descriptor layout (approx): { allocated_ptr, aligned_ptr, offset, sizes..., strides... }
// We only need the aligned_ptr, which is at index 1 of the pointer array.
static inline float* extract_ptr(void* descriptor) {
    float** ptrs = (float**)descriptor;
    return ptrs[1];
}

// --- Implementation Logic (Internal) ---

void rc_dense_matmul_bn_nn(const float* X, const float* W, float* Y, int B, int N) {
  for (int b = 0; b < B; ++b) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < N; ++k) {
        acc += X[b*N + k] * W[k*N + j];
      }
      Y[b*N + j] = acc;
    }
  }
}

void rc_dense_matmul_bd_dn(const float* U, const float* Win, float* Y, int B, int Din, int N) {
  for (int b = 0; b < B; ++b) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < Din; ++k) {
        acc += U[b*Din + k] * Win[k*N + j];
      }
      Y[b*N + j] = acc;
    }
  }
}

void rc_tanh_vec(float* X, int len) {
  for (int i = 0; i < len; ++i) X[i] = tanhf(X[i]);
}

void rc_leaky_mix(const float* x, const float* act, float* out, int len, float leak) {
  float one_minus = 1.0f - leak;
  for (int i = 0; i < len; ++i) out[i] = one_minus * x[i] + leak * act[i];
}

static inline float rc_idxhash(int i, int j) {
  uint64_t x = (uint64_t)(i + 1) * 1315423911ULL ^ (uint64_t)(j + 1) * 2654435761ULL;
  x ^= (x >> 13); x *= 0x5bd1e995ULL; x ^= (x >> 15);
  uint32_t m = (uint32_t)(x & 0x00FFFFFFu);
  float u01 = (float)m / 16777215.0f;
  return 2.0f * u01 - 1.0f;
}

// --- Exposed ABI Wrappers (called from MLIR) ---
// MLIR Unranked MemRef passes: (int64_t rank, void* descriptor_ptr)

void rc_fill2d(int64_t rank, void* desc, int d0, int d1, float scale) {
    float* out = extract_ptr(desc);
    for (int i = 0; i < d0; ++i) {
        for (int j = 0; j < d1; ++j) {
            out[i*d1 + j] = rc_idxhash(i, j) * scale;
        }
    }
}

void rc_dense_step(
    int64_t rW, void* dW,
    int64_t rWin, void* dWin,
    int64_t ru, void* du,
    int64_t rx, void* dx,
    int64_t rx2, void* dx2,
    int64_t rtmp1, void* dtmp1,
    int64_t rtmp2, void* dtmp2,
    int64_t rpre,  void* dpre,
    int B, int N, int Din, float leak)
{
    float* W = extract_ptr(dW);
    float* Win = extract_ptr(dWin);
    float* u = extract_ptr(du);
    float* x = extract_ptr(dx);
    float* x_next = extract_ptr(dx2);

    float* tmp1 = extract_ptr(dtmp1);
    float* tmp2 = extract_ptr(dtmp2);
    float* pre  = extract_ptr(dpre);

    rc_dense_matmul_bn_nn(x, W, tmp1, B, N);
    rc_dense_matmul_bd_dn(u, Win, tmp2, B, Din, N);

    int len = B * N;
    for (int i = 0; i < len; ++i) pre[i] = tmp1[i] + tmp2[i];
    rc_tanh_vec(pre, len);
    rc_leaky_mix(x, pre, x_next, len, leak);
}

#ifdef __cplusplus
}
#endif