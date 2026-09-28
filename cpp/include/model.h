#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

class OnnxModel {
public:
    explicit OnnxModel(const std::string& model_path);
    std::vector<float> predict(const std::vector<float>& features,
                               std::size_t batch_size, std::size_t num_features);
    const std::string& feature_header() const { return feature_header_; }
    std::size_t num_features() const { return num_features_; }

private:
    // Declaration order ensures the session dies before the environment.
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "conversion"};
    Ort::SessionOptions options_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::string feature_header_;
    std::size_t num_features_{};
};
