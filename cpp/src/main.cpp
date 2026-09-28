#include "benchmark.h"
#include "model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t number(const std::string& text, bool allow_zero = false) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected an unsigned integer: " + text);
    const auto value = std::stoull(text);
    if ((!allow_zero && value == 0) || value > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Invalid count: " + text);
    return static_cast<std::size_t>(value);
}

std::vector<float> read_csv(const std::string& path, const OnnxModel& model) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open input CSV: " + path);
    std::string line;
    std::getline(stream, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != model.feature_header())
        throw std::runtime_error("CSV header/order must exactly match model feature_names metadata.");
    std::vector<float> values;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.back() == ',') throw std::runtime_error("Empty CSV row or field.");
        std::istringstream row(line);
        std::string cell;
        std::size_t columns = 0;
        while (std::getline(row, cell, ',')) {
            std::size_t consumed = 0;
            const float value = std::stof(cell, &consumed);
            if (consumed != cell.size() || !std::isfinite(value))
                throw std::runtime_error("CSV must contain finite numeric feature values.");
            values.push_back(value);
            ++columns;
        }
        if (columns != model.num_features()) throw std::runtime_error("Incorrect CSV column count.");
    }
    if (values.empty()) throw std::runtime_error("CSV has no feature rows.");
    return values;
}
}

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> args{{"--model", "artifacts/model.onnx"},
            {"--input", "artifacts/test_features.csv"}, {"--output", "artifacts/cpp_predictions.csv"},
            {"--metrics", "artifacts/cpp_benchmark.json"}, {"--batch-size", "256"},
            {"--iterations", "100"}, {"--warmup", "10"}};
        for (int i = 1; i < argc; i += 2) {
            if (std::string(argv[i]) == "--help") {
                std::cout << "predict --model MODEL --input CSV --output CSV --metrics JSON "
                             "--batch-size 256 --iterations 100 --warmup 10\n";
                return 0;
            }
            if (i + 1 >= argc || !args.count(argv[i])) throw std::invalid_argument("Unknown or missing CLI option.");
            args[argv[i]] = argv[i + 1];
        }
        const auto batch_size = number(args.at("--batch-size"));
        const auto iterations = number(args.at("--iterations"));
        const auto warmup = number(args.at("--warmup"), true);
        OnnxModel model(args.at("--model"));
        const auto features = read_csv(args.at("--input"), model);
        const auto columns = model.num_features();
        const auto rows = features.size() / columns;
        if (batch_size > rows) throw std::invalid_argument("batch-size must not exceed input rows.");
        std::ofstream predictions(args.at("--output"));
        if (!predictions) throw std::runtime_error("Cannot open prediction output.");
        predictions << "probability\n" << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (std::size_t offset = 0; offset < rows; offset += batch_size) {
            const auto count = std::min(batch_size, rows - offset);
            const std::vector<float> batch(features.begin() + offset * columns,
                                           features.begin() + (offset + count) * columns);
            for (const auto probability : model.predict(batch, count, columns)) {
                if (!std::isfinite(probability) || probability < 0 || probability > 1)
                    throw std::runtime_error("Model returned an invalid probability.");
                predictions << probability << '\n';
            }
        }
        if (!predictions) throw std::runtime_error("Failed to write predictions.");
        const std::vector<float> batch(features.begin(), features.begin() + batch_size * columns);
        const auto stats = benchmark(model, batch, batch_size, columns, iterations, warmup);
        std::ostringstream json;
        json << std::setprecision(12) << "{\"mean_ms\":" << stats.mean_ms
             << ",\"p50_ms\":" << stats.p50_ms << ",\"p95_ms\":" << stats.p95_ms
             << ",\"samples_per_sec\":" << stats.samples_per_sec << ",\"batch_size\":" << batch_size
             << ",\"iterations\":" << iterations << ",\"warmup\":" << warmup
             << ",\"onnxruntime_version\":\"" << Ort::GetVersionString() << "\"}\n";
        std::ofstream metrics(args.at("--metrics"));
        metrics << json.str();
        if (!metrics) throw std::runtime_error("Cannot write benchmark metrics.");
        std::cout << json.str();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
