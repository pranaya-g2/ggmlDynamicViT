// graph.cpp
#include "dynvit-impl.h"

#include <stdexcept>
#include <vector>

namespace dynvit {

static ggml_context * new_compute_ctx() {
    ggml_init_params params = {
        16 * 1024 * 1024,
        nullptr,
        true
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("Failed to create GGML compute context");

    return ctx;
}

using scheduler_ptr = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;

static scheduler_ptr run_graph(model & m, ggml_context * ctx, ggml_cgraph * graph, ggml_tensor * input, const float * input_data) {
    ggml_backend_t backends[] = {m.backend};

    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true
    );

    if (!sched) throw std::runtime_error("Failed to create backend scheduler");

    scheduler_ptr owner(sched, ggml_backend_sched_free);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) throw std::runtime_error("GGML graph allocation failed");
    ggml_backend_tensor_set(input, input_data, 0, ggml_nbytes(input));

    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GGML graph compute failed");
    }

    ggml_backend_sched_synchronize(sched);
    return owner;
}

ggml_tensor * build_patch_embed(ggml_context * ctx, const model & m, ggml_tensor * image) {
    // Keep im2col in F32: ggml_conv_2d otherwise rounds patches to F16.
    auto * patches = ggml_im2col(ctx, m.patch_embed_w, image, 16, 16, 0, 0, 1, 1, true, GGML_TYPE_F32);
    auto * weights = ggml_reshape_2d(ctx, m.patch_embed_w, 16 * 16 * 3, 384);
    patches = ggml_reshape_2d(ctx, patches, 16 * 16 * 3, 196);
    ggml_tensor * x = ggml_mul_mat(ctx, weights, patches);

    ggml_tensor * patch_bias = ggml_repeat_4d(ctx, m.patch_embed_b, 384, 196, 1, 1);
    x = ggml_add(ctx, x, patch_bias);

    return x;
}

bool run_stage0(model & m, const float * input_chw, std::vector<float> & x_out, std::vector<float> & scores_out) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(new_compute_ctx(), ggml_free);
    ggml_context * ctx = context.get();
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);

    ggml_tensor * image = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 224, 224, 3, 1);
    ggml_set_input(image);

    ggml_tensor * x = build_patch_embed(ctx, m, image);

    ggml_tensor * cls = ggml_reshape_2d(ctx, m.cls_token, 384, 1);
    x = ggml_concat(ctx, cls, x, 1);

    ggml_tensor * pos = ggml_reshape_2d(ctx, m.pos_embed, 384, 197);
    x = ggml_add(ctx, x, pos);

    for (int i = 0; i < 3; ++i) x = build_block(ctx, m, i, x, 197);

    ggml_tensor * patches = ggml_view_2d(ctx, x, 384, 196, x->nb[1], x->nb[1]);
    ggml_tensor * scores = build_predictor(ctx, m, 0, patches, 196);

    ggml_set_output(x);
    ggml_set_output(scores);
    ggml_build_forward_expand(graph, x);
    ggml_build_forward_expand(graph, scores);

    auto scheduler = run_graph(m, ctx, graph, image, input_chw);

    x_out.resize(384 * 197);
    scores_out.resize(2 * 196);

    ggml_backend_tensor_get(x, x_out.data(), 0, x_out.size() * sizeof(float));
    ggml_backend_tensor_get(scores, scores_out.data(), 0, scores_out.size() * sizeof(float));

    return true;
}

bool run_middle_stage(model & m, const std::vector<float> & input, int32_t n_tokens, int begin_block, int end_block, int predictor_index, std::vector<float> & x_out, std::vector<float> & scores_out) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(new_compute_ctx(), ggml_free);
    ggml_context * ctx = context.get();
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 384, n_tokens);
    ggml_set_input(x);
    ggml_tensor * input_tensor = x;

    for (int i = begin_block; i <= end_block; ++i) x = build_block(ctx, m, i, x, n_tokens);

    int32_t n_patches = n_tokens - 1;

    ggml_tensor * patches = ggml_view_2d(ctx, x, 384, n_patches, x->nb[1], x->nb[1]);
    ggml_tensor * scores = build_predictor(ctx, m, predictor_index, patches, n_patches);

    ggml_set_output(x);
    ggml_set_output(scores);
    ggml_build_forward_expand(graph, x);
    ggml_build_forward_expand(graph, scores);

    auto scheduler = run_graph(m, ctx, graph, input_tensor, input.data());

    x_out.resize(384 * n_tokens);
    scores_out.resize(2 * n_patches);

    ggml_backend_tensor_get(x, x_out.data(), 0, x_out.size() * sizeof(float));
    ggml_backend_tensor_get(scores, scores_out.data(), 0, scores_out.size() * sizeof(float));

    return true;
}

bool run_final_stage(model & m, const std::vector<float> & input, int32_t n_tokens, std::vector<float> & logits) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(new_compute_ctx(), ggml_free);
    ggml_context * ctx = context.get();
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 384, n_tokens);
    ggml_set_input(x);
    ggml_tensor * input_tensor = x;

    for (int i = 9; i < 12; ++i) x = build_block(ctx, m, i, x, n_tokens);

    x = layer_norm(ctx, x, m.final_norm_w, m.final_norm_b, m.hp.norm_eps);

    ggml_tensor * cls = ggml_view_2d(ctx, x, 384, 1, x->nb[1], 0);
    ggml_tensor * out = linear(ctx, cls, m.head_w, m.head_b);

    ggml_build_forward_expand(graph, out);
    auto scheduler = run_graph(m, ctx, graph, input_tensor, input.data());
    logits.resize(1000);
    ggml_backend_tensor_get(out, logits.data(), 0, logits.size() * sizeof(float));

    return true;
}

}