#include "benchmark.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>

namespace {
double percentile(const std::vector<double>& sorted, double fraction) {
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const auto low = static_cast<std::size_t>(position);
    const auto high = std::min(low + 1, sorted.size() - 1);
    return sorted[low] + (position - static_cast<double>(low)) * (sorted[high] - sorted[low]);
}
}

BenchmarkResult benchmark(OnnxModel& model, const std::vector<float>& features,
                          std::size_t batch_size, std::size_t num_features,
                          std::size_t iterations, std::size_t warmup) {
    if (iterations == 0) throw std::invalid_argument("iterations must be positive");
    for (std::size_t i = 0; i < warmup; ++i) model.predict(features, batch_size, num_features);
    std::vector<double> milliseconds;
    milliseconds.reserve(iterations);
    for (std::size_t i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const auto predictions = model.predict(features, batch_size, num_features);
        const auto end = std::chrono::steady_clock::now();
        milliseconds.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    const double mean = std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) /
                        static_cast<double>(iterations);
    std::sort(milliseconds.begin(), milliseconds.end());
    return {mean, percentile(milliseconds, 0.5), percentile(milliseconds, 0.95),
            static_cast<double>(batch_size) * 1000.0 / mean};
}
