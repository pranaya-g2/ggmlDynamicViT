#include "dynvit-impl.h"

#include <cmath>

namespace dynvit {

ggml_tensor * build_attention(ggml_context * ctx, const model & m, int block_index, ggml_tensor * x, int32_t n_tokens) {
    const auto & b = m.blocks[block_index];

    const int64_t C = m.hp.embed_dim;
    const int64_t H = m.hp.num_heads;
    const int64_t D = m.hp.head_dim;

    // QKV = X Wqkv^T + b
    ggml_tensor * qkv = linear(ctx, x, b.qkv_w, b.qkv_b);

    qkv = ggml_reshape_4d(ctx, qkv, D, H, 3, n_tokens);

    // Views into the QKV tensor.
    ggml_tensor * q = ggml_view_3d(ctx, qkv, D, H, n_tokens, qkv->nb[1], qkv->nb[3], 0);
    ggml_tensor * k = ggml_view_3d(ctx, qkv, D, H, n_tokens, qkv->nb[1], qkv->nb[3], qkv->nb[2]);
    ggml_tensor * v = ggml_view_3d(ctx, qkv, D, H, n_tokens, qkv->nb[1], qkv->nb[3], 2 * qkv->nb[2]);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // scores = Q K^T
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // softmax((QK^T) / sqrt(D))
    ggml_tensor * attn = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);

    // V needs shape [N, D, H] for this matmul.
    ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, v));

    // attn at V
    ggml_tensor * out = ggml_mul_mat(ctx, vt, attn);
    out = ggml_permute(ctx, out, 0, 2, 1, 3);

    // Make heads contiguous, then flatten:
    out = ggml_cont_3d(ctx, out, D, H, n_tokens);
    out = ggml_reshape_2d(ctx, out, C, n_tokens);

    out = linear(ctx, out, b.attn_proj_w, b.attn_proj_b);
    return out;
}

}
