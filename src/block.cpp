#include "dynvit-impl.h"

namespace dynvit {

ggml_tensor * build_block(ggml_context * ctx, const model & m, int block_index, ggml_tensor * x, int32_t n_tokens) {
    const auto & b = m.blocks[block_index];

    ggml_tensor * y = layer_norm(ctx, x, b.norm1_w, b.norm1_b, m.hp.norm_eps);
    y = build_attention(ctx, m, block_index, y, n_tokens);
    x = ggml_add(ctx, x, y);

    y = layer_norm(ctx, x, b.norm2_w, b.norm2_b, m.hp.norm_eps);
    y = linear(ctx, y, b.mlp_fc1_w, b.mlp_fc1_b);
    y = gelu(ctx, y);
    y = linear(ctx, y, b.mlp_fc2_w, b.mlp_fc2_b);

    return ggml_add(ctx, x, y);
}

}