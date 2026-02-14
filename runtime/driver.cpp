#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int32_t bench();

static uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char** argv) {
    int warmup = 5;
    int runs = 30;
    if (argc >= 2) warmup = std::atoi(argv[1]);
    if (argc >= 3) runs = std::atoi(argv[2]);

    // warmup
    for (int i = 0; i < warmup; ++i) (void)bench();

    uint64_t best = UINT64_MAX;
    uint64_t sum = 0;
    int32_t sink = 0;

    for (int i = 0; i < runs; ++i) {
        uint64_t t0 = now_ns();
        sink ^= bench();
        uint64_t t1 = now_ns();
        uint64_t dt = t1 - t0;
        if (dt < best) best = dt;
        sum += dt;
    }

    double avg = (runs > 0) ? (double)sum / (double)runs : 0.0;
    std::printf("best_ns=%llu,avg_ns=%.2f,sink=%d\n", (unsigned long long)best,
                avg, (int)sink);
    return 0;
}
