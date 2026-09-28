#include "model.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>

OnnxModel::OnnxModel(const std::string& model_path) {
    options_.SetIntraOpNumThreads(1);
    options_.SetInterOpNumThreads(1);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // filesystem::path provides the native wide string required by Windows ORT.
    session_ = Ort::Session(env_, std::filesystem::path(model_path).c_str(), options_);
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
        throw std::runtime_error("Expected one features input and one probabilities output.");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = session_.GetInputNameAllocated(0, allocator);
    const auto output_name = session_.GetOutputNameAllocated(0, allocator);
    if (std::string(input_name.get()) != "features" || std::string(output_name.get()) != "probabilities") {
        throw std::runtime_error("Model does not match the exported feature/probability contract.");
    }
    const auto input_info = session_.GetInputTypeInfo(0);
    const auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
    const auto shape = tensor_info.GetShape();
    if (shape.size() != 2 || shape[1] <= 0 ||
        tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Expected float32 [batch, num_features] input.");
    }
    num_features_ = static_cast<std::size_t>(shape[1]);
    const auto header = session_.GetModelMetadata().LookupCustomMetadataMapAllocated("feature_names", allocator);
    if (!header || std::string(header.get()).empty()) {
        throw std::runtime_error("Missing feature_names metadata; use export.export_onnx.");
    }
    feature_header_ = header.get();
}

std::vector<float> OnnxModel::predict(const std::vector<float>& features,
                                    std::size_t batch_size, std::size_t num_features) {
    if (!batch_size || num_features != num_features_ ||
        batch_size > std::numeric_limits<std::size_t>::max() / num_features ||
        features.size() != batch_size * num_features) {
        throw std::invalid_argument("Feature buffer does not match the model/batch shape.");
    }
    const std::array<int64_t, 2> shape{static_cast<int64_t>(batch_size), static_cast<int64_t>(num_features)};
    // ORT's external-buffer API is mutable, but inference only reads this tensor.
    auto input = Ort::Value::CreateTensor<float>(memory_, const_cast<float*>(features.data()),
                                                features.size(), shape.data(), shape.size());
    const char* input_names[] = {"features"};
    const char* output_names[] = {"probabilities"};
    auto output = session_.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
    const auto info = output[0].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        info.GetShape() != std::vector<int64_t>{static_cast<int64_t>(batch_size), 1}) {
        throw std::runtime_error("Expected float32 [batch, 1] probabilities.");
    }
    const float* values = output[0].GetTensorData<float>();
    return {values, values + batch_size};
}
