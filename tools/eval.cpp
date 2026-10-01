#include "dynvit-impl.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <iostream>

namespace fs = std::filesystem;

static int evaluate(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " model.gguf dataset_root\n";
        return 1;
    }

    if (!fs::is_directory(argv[2])) throw std::runtime_error("Dataset root is not a directory");
    auto model = dynvit::load_model(argv[1]);
    if (!model) throw std::runtime_error("Failed to load model");
    dynvit::preprocess_params pp;

    int total = 0;
    int correct1 = 0;
    int correct5 = 0;

    for (const auto & class_entry : fs::directory_iterator(argv[2])) {
        if (!class_entry.is_directory()) continue;

        const std::string name = class_entry.path().filename().string();
        if (name.empty() || name.find_first_not_of("0123456789") != std::string::npos) {
            throw std::runtime_error("Expected zero-based numeric class directory: " + name);
        }
        int label = std::stoi(name);
        if (label < 0 || label >= model->hp.num_classes) throw std::runtime_error("Class label out of range: " + name);

        for (const auto & image_entry : fs::directory_iterator(class_entry.path())) {
            if (!image_entry.is_regular_file()) continue;

            std::vector<float> input;

            if (!dynvit::preprocess_image(image_entry.path().string(), pp, input)) {
                std::cerr << "Failed to preprocess: " << image_entry.path() << "\n";
                continue;
            }

            dynvit::inference_result result;

            if (!dynvit::run_inference(*model, input.data(), result)) {
                std::cerr << "Inference failed: " << image_entry.path() << "\n";
                continue;
            }

            if (result.logits.size() != static_cast<size_t>(model->hp.num_classes) || result.logits.size() < 5 ||
                !std::all_of(result.logits.begin(), result.logits.end(), [](float v) { return std::isfinite(v); })) {
                throw std::runtime_error("Invalid logits for: " + image_entry.path().string());
            }

            int pred1 = dynvit::argmax(result.logits);
            auto pred5 = dynvit::topk(result.logits, 5);

            if (pred1 == label) correct1++;
            if (dynvit::contains(pred5, label)) correct5++;

            total++;

            if (total % 100 == 0) {
                std::cout
                    << total
                    << " images | top1 "
                    << 100.0 * correct1 / total
                    << "% | top5 "
                    << 100.0 * correct5 / total
                    << "%\n";
            }
        }
    }

    if (total == 0) {
        std::cerr << "No images evaluated\n";
        return 1;
    }

    std::cout << "\nFinal\n";
    std::cout << "images: " << total << "\n";
    std::cout << "top1: " << 100.0 * correct1 / total << "%\n";
    std::cout << "top5: " << 100.0 * correct5 / total << "%\n";

    return 0;
}

int main(int argc, char ** argv) {
    try {
        return evaluate(argc, argv);
    } catch (const std::exception & e) {
        std::cerr << "Evaluation failed: " << e.what() << "\n";
        return 1;
    }
}
