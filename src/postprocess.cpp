#include "dynvit-impl.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace dynvit {

int argmax(const std::vector<float> & logits) {
    if (logits.empty()) throw std::invalid_argument("argmax requires nonempty logits");
    return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

std::vector<int> topk(const std::vector<float> & logits, int k) {
    if (k < 0 || static_cast<size_t>(k) > logits.size()) throw std::invalid_argument("topk requires 0 <= k <= logits.size()");
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b) {
            return logits[a] > logits[b];
        });

    idx.resize(k);
    return idx;
}

bool contains(const std::vector<int> & v, int x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

}