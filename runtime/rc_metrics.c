#include <math.h>
#include <stdint.h>

typedef struct {
    void* allocated;
    float* aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
} MemRef2D_f32;

static inline MemRef2D_f32* as2d(void* desc) { return (MemRef2D_f32*)desc; }

double rc_rmse2d(int64_t rankA, void* descA, int64_t rankB, void* descB,
                 int64_t d0, int64_t d1) {
    (void)rankA;
    (void)rankB;
    MemRef2D_f32* A = as2d(descA);
    MemRef2D_f32* B = as2d(descB);

    double sse = 0.0;
    double n = (double)(d0 * d1);

    float* ap = A->aligned + A->offset;
    float* bp = B->aligned + B->offset;
    int64_t as0 = A->strides[0], as1 = A->strides[1];
    int64_t bs0 = B->strides[0], bs1 = B->strides[1];

    for (int64_t i = 0; i < d0; i++) {
        for (int64_t j = 0; j < d1; j++) {
            float da = ap[i * as0 + j * as1];
            float db = bp[i * bs0 + j * bs1];
            double diff = (double)da - (double)db;
            sse += diff * diff;
        }
    }
    return sqrt(sse / n);
}

double rc_rsquared2d(int64_t rankA, void* descA, int64_t rankB, void* descB,
                     int64_t d0, int64_t d1) {
    (void)rankA;
    (void)rankB;
    MemRef2D_f32* Y = as2d(descB);  // true
    MemRef2D_f32* P = as2d(descA);  // pred

    float* yp = Y->aligned + Y->offset;
    float* pp = P->aligned + P->offset;
    int64_t ys0 = Y->strides[0], ys1 = Y->strides[1];
    int64_t ps0 = P->strides[0], ps1 = P->strides[1];

    double n = (double)(d0 * d1);

    // mean of y
    double mean = 0.0;
    for (int64_t i = 0; i < d0; i++)
        for (int64_t j = 0; j < d1; j++) mean += (double)yp[i * ys0 + j * ys1];
    mean /= n;

    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (int64_t i = 0; i < d0; i++) {
        for (int64_t j = 0; j < d1; j++) {
            double y = (double)yp[i * ys0 + j * ys1];
            double p = (double)pp[i * ps0 + j * ps1];
            double diff = y - p;
            ss_res += diff * diff;
            double t = y - mean;
            ss_tot += t * t;
        }
    }
    if (ss_tot == 0.0) return 0.0;
    return 1.0 - (ss_res / ss_tot);
}
