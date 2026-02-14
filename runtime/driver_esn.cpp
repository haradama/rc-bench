#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int32_t bench();
extern "C" int32_t infer_store();
extern "C" double rmse();
extern "C" double r2();

static uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char** argv) {
    int warmup = (argc > 1) ? std::atoi(argv[1]) : 3;
    int runs = (argc > 2) ? std::atoi(argv[2]) : 10;

    for (int i = 0; i < warmup; i++) (void)bench();

    uint64_t best = UINT64_MAX;
    uint64_t sum = 0;
    int32_t sink = 0;

    for (int i = 0; i < runs; i++) {
        auto t0 = now_ns();
        sink ^= bench();
        auto t1 = now_ns();
        auto dt = t1 - t0;
        best = std::min(best, dt);
        sum += dt;
    }

    // correctness (not timed)
    (void)infer_store();
    double e = rmse();
    double r = r2();

    double avg = (runs > 0) ? (double)sum / (double)runs : 0.0;
    std::printf("best_ns=%llu,avg_ns=%.2f,sink=%d,rmse=%.10g,r2=%.10g\n",
                (unsigned long long)best, avg, (int)sink, e, r);
    return 0;
}
