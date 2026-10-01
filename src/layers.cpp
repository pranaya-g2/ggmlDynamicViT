#include "dynvit-impl.h"

namespace dynvit {

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias) {
    ggml_tensor * y = ggml_mul_mat(ctx, weight, x);

    if (bias) {
        ggml_tensor * b = ggml_repeat_4d(ctx, bias, y->ne[0], y->ne[1], y->ne[2], y->ne[3]);
        y = ggml_add(ctx, y, b);
    }

    return y;
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias, float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);

    ggml_tensor * w = ggml_repeat_4d(ctx, weight, y->ne[0], y->ne[1], y->ne[2], y->ne[3]);
    ggml_tensor * b = ggml_repeat_4d(ctx, bias,   y->ne[0], y->ne[1], y->ne[2], y->ne[3]);

    y = ggml_mul(ctx, y, w);
    y = ggml_add(ctx, y, b);

    return y;
}

ggml_tensor * gelu(ggml_context * ctx, ggml_tensor * x) {
    return ggml_gelu_erf(ctx, x);
}

} 