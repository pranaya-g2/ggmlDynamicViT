#include "dynvit-impl.h"

namespace dynvit {

ggml_tensor * build_predictor(ggml_context * ctx, const model & m, int predictor_index, ggml_tensor * x_patch, int32_t n_patch_tokens) {
    const auto & p = m.predictors[predictor_index];

    const int64_t C = m.hp.embed_dim;
    const int64_t C2 = C / 2;

    ggml_tensor * z = layer_norm(ctx, x_patch, p.norm_w, p.norm_b, m.hp.norm_eps);
    z = linear(ctx, z, p.in_w, p.in_b);
    z = gelu(ctx, z);

    ggml_tensor * local = ggml_view_2d(ctx, z, C2, n_patch_tokens, z->nb[1], 0);
    ggml_tensor * global = ggml_view_2d(ctx, z, C2, n_patch_tokens, z->nb[1], C2 * z->nb[0]);

    global = ggml_transpose(ctx, global);
    // GGML mean reduces contiguous rows; transpose alone is only a view.
    global = ggml_mean(ctx, ggml_cont(ctx, global));
    global = ggml_transpose(ctx, global);
    global = ggml_repeat_4d(ctx, global, C2, n_patch_tokens, 1, 1);

    ggml_tensor * h = ggml_concat(ctx, local, global, 0);

    h = linear(ctx, h, p.fc1_w, p.fc1_b);
    h = gelu(ctx, h);

    h = linear(ctx, h, p.fc2_w, p.fc2_b);
    h = gelu(ctx, h);

    ggml_tensor * logits = linear(ctx, h, p.fc3_w, p.fc3_b);
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    return ggml_log(ctx, probs);
}

} // namespace dynvit