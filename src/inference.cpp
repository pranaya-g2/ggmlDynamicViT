// inference.cpp
#include "dynvit-impl.h"

#include <vector>

namespace dynvit {

bool run_inference(model & m, const float * input_chw, inference_result & result) {
    result.logits.clear();
    if (!input_chw) return false;
    std::vector<float> x, scores;

    if (!run_stage0(m, input_chw, x, scores)) return false;

    auto keep0 = select_topk_tokens(scores.data(), 196, 137);
    x = compact_tokens(x, 197, 384, keep0);

    result.token_counts[0] = 197;
    result.token_counts[1] = 138;

    if (!run_middle_stage(m, x, 138, 3, 5, 1, x, scores)) return false;

    auto keep1 = select_topk_tokens(scores.data(), 137, 96);
    x = compact_tokens(x, 138, 384, keep1);

    result.token_counts[2] = 97;

    if (!run_middle_stage(m, x, 97, 6, 8, 2, x, scores)) return false;

    auto keep2 = select_topk_tokens(scores.data(), 96, 67);
    x = compact_tokens(x, 97, 384, keep2);

    result.token_counts[3] = 68;

    return run_final_stage(m, x, 68, result.logits);
}

}