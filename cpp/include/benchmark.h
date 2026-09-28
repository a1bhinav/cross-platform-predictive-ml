#pragma once

#include "model.h"

struct BenchmarkResult {
    double mean_ms;
    double p50_ms;
    double p95_ms;
    double samples_per_sec;
};

BenchmarkResult benchmark(OnnxModel& model, const std::vector<float>& features,
                          std::size_t batch_size, std::size_t num_features,
                          std::size_t iterations, std::size_t warmup);
