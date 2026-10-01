// prune.cpp
#include "dynvit-impl.h"

#include <algorithm>
#include <numeric>

namespace dynvit {

std::vector<int32_t> select_topk_tokens(const float * scores, int32_t n_tokens, int32_t n_keep) {
    std::vector<int32_t> idx(n_tokens);
    std::iota(idx.begin(), idx.end(), 0);

    std::partial_sort(idx.begin(), idx.begin() + n_keep, idx.end(),
        [&](int32_t a, int32_t b) {
            return scores[2 * a] > scores[2 * b];
        });

    idx.resize(n_keep);
    return idx;
}

std::vector<float> compact_tokens(const std::vector<float> & x, int32_t n_tokens, int32_t embed_dim, const std::vector<int32_t> & keep) {
    std::vector<float> out(embed_dim * (keep.size() + 1));

    std::copy(x.begin(), x.begin() + embed_dim, out.begin());

    for (size_t j = 0; j < keep.size(); ++j) {
        int32_t src_token = keep[j] + 1;

        const float * src = x.data() + src_token * embed_dim;
        float * dst = out.data() + (j + 1) * embed_dim;

        std::copy(src, src + embed_dim, dst);
    }

    return out;
}

} // namespace dynvit