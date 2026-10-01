#ifndef DYNVIT_IMPL_H
#define DYNVIT_IMPL_H

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <memory>

#include "../ggml/include/ggml.h"
#include "../ggml/include/ggml-backend.h"

namespace dynvit {
// Constants
constexpr int DV_IMAGE_SIZE       = 224;
constexpr int DV_PATCH_SIZE       = 16;
constexpr int DV_IN_CHANNELS      = 3;
constexpr int DV_EMBED_DIM        = 384;
constexpr int DV_NUM_HEADS        = 6;
constexpr int DV_HEAD_DIM         = 64;
constexpr int DV_NUM_BLOCKS       = 12;
constexpr int DV_MLP_DIM          = 1536;
constexpr int DV_NUM_CLASSES      = 1000;
constexpr int DV_NUM_PATCHES      = 196; // 14 * 14
constexpr int DV_NUM_TOKENS       = 197; // 196 + CLS
constexpr int DV_NUM_PREDICTORS   = 3;
constexpr float DV_LAYER_NORM_EPS = 1e-6f;

// Preprocessing
// 8-bit RGB image, interleaved HWC (row-major, 3 bytes per pixel).
struct image_u8 {
    int w = 0;
    int h = 0;
    std::vector<uint8_t> data;
};
struct preprocess_params {
    int resize_short = 256; // int(224 / 0.875)
    int crop         = 224;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
};

// Decode JPEG/PNG to RGB, expanding grayscale to 3 channels as Pillow does.
bool load_image_rgb(const std::string & path, image_u8 & out);

// Bit-exact port of Pillow Image.resize(..., Image.BICUBIC).
image_u8 resize_bicubic_pil(const image_u8 & in, int out_w, int out_h);

// torchvision CenterCrop with round-half-to-even offsets, like Python's round().
image_u8 center_crop(const image_u8 & in, int crop_w, int crop_h);

// Transform decoded pixels to CHW [3, 224, 224], indexed by c * H * W + h * W + w; GGML ne = [224, 224, 3].
void preprocess_u8(const image_u8 & in, const preprocess_params & p, float * out);

// load_image_rgb + preprocess_u8.
bool preprocess_image(const std::string & path, const preprocess_params & p, std::vector<float> & out);

// Model hyperparameters
struct hparams {
    int32_t image_size   = DV_IMAGE_SIZE;
    int32_t patch_size   = DV_PATCH_SIZE;
    int32_t in_channels  = DV_IN_CHANNELS;
    int32_t embed_dim    = DV_EMBED_DIM;
    int32_t num_heads    = DV_NUM_HEADS;
    int32_t head_dim     = DV_HEAD_DIM;
    int32_t num_blocks   = DV_NUM_BLOCKS;
    int32_t mlp_dim      = DV_MLP_DIM;
    int32_t num_classes  = DV_NUM_CLASSES;
    int32_t num_patches  = DV_NUM_PATCHES;
    float norm_eps       = DV_LAYER_NORM_EPS;
    std::array<int32_t, DV_NUM_PREDICTORS> prune_layers = {3, 6, 9};
    std::array<float, DV_NUM_PREDICTORS> keep_ratios = {0.7f, 0.49f, 0.343f};
};

// Transformer block weights
struct block_weights {
    // LayerNorm before attention.
    ggml_tensor * norm1_w = nullptr;
    ggml_tensor * norm1_b = nullptr;

    // Combined QKV projection: Linear(384, 1152).
    ggml_tensor * qkv_w = nullptr;
    ggml_tensor * qkv_b = nullptr;

    // Attention output projection: Linear(384, 384).
    ggml_tensor * attn_proj_w = nullptr;
    ggml_tensor * attn_proj_b = nullptr;

    // LayerNorm before MLP.
    ggml_tensor * norm2_w = nullptr;
    ggml_tensor * norm2_b = nullptr;

    // MLP: 384 -> 1536 -> 384
    ggml_tensor * mlp_fc1_w = nullptr;
    ggml_tensor * mlp_fc1_b = nullptr;
    ggml_tensor * mlp_fc2_w = nullptr;
    ggml_tensor * mlp_fc2_b = nullptr;
};

// DynamicViT PredictorLG weights
struct predictor_weights {
    // First transform: LayerNorm(384), Linear(384, 384), GELU.
    ggml_tensor * norm_w = nullptr;
    ggml_tensor * norm_b = nullptr;
    ggml_tensor * in_w = nullptr;
    ggml_tensor * in_b = nullptr;

    // Predictor MLP after local/global concatenation: 384 -> 192 -> 96 -> 2
    ggml_tensor * fc1_w = nullptr;
    ggml_tensor * fc1_b = nullptr;
    ggml_tensor * fc2_w = nullptr;
    ggml_tensor * fc2_b = nullptr;
    ggml_tensor * fc3_w = nullptr;
    ggml_tensor * fc3_b = nullptr;
};

// Entire DynamicViT model
struct model {
    hparams hp;

    // GGML ownership
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * patch_embed_w = nullptr;
    ggml_tensor * patch_embed_b = nullptr;
    // Learned CLS token: [1, 384].
    ggml_tensor * cls_token = nullptr;
    // Learned positional embedding: [197, 384].
    ggml_tensor * pos_embed = nullptr;
    // Transformer
    std::array<block_weights, DV_NUM_BLOCKS> blocks;
    // Dynamic token predictors
    std::array<predictor_weights, DV_NUM_PREDICTORS> predictors;
    // Final normalization
    ggml_tensor * final_norm_w = nullptr;
    ggml_tensor * final_norm_b = nullptr;
    // Classification head
    // Linear: 384 -> 1000
    ggml_tensor * head_w = nullptr;
    ggml_tensor * head_b = nullptr;
};

// Runtime / inference structures
struct token_state {
    ggml_tensor * x = nullptr;
    int32_t n_tokens = 0;
};

// Result of one predictor stage.
struct prune_result {
    std::vector<int32_t> indices;
    // Number of retained image tokens, excluding CLS.
    int32_t n_keep = 0;
};

// Raw inference result.
struct inference_result {
    // 1000 ImageNet logits.
    std::vector<float> logits;
    std::array<int32_t, 4> token_counts = {197, 138, 97, 68};
};

// GGUF metadata keys
namespace gguf_key {
constexpr const char * ARCHITECTURE = "general.architecture";
constexpr const char * IMAGE_SIZE = "dynamicvit.image_size";
constexpr const char * PATCH_SIZE = "dynamicvit.patch_size";
constexpr const char * IN_CHANNELS = "dynamicvit.in_channels";
constexpr const char * EMBED_DIM = "dynamicvit.embedding_length";
constexpr const char * NUM_HEADS = "dynamicvit.attention.head_count";
constexpr const char * NUM_BLOCKS = "dynamicvit.block_count";
constexpr const char * MLP_DIM = "dynamicvit.feed_forward_length";
constexpr const char * NUM_CLASSES = "dynamicvit.class_count";
constexpr const char * NORM_EPS = "dynamicvit.layer_norm_epsilon";
constexpr const char * PRUNE_LAYERS = "dynamicvit.prune_layers";
constexpr const char * KEEP_RATIOS = "dynamicvit.keep_ratios";
}

// Tensor names
namespace tensor_name {
constexpr const char * PATCH_EMBED_W = "patch_embed.proj.weight";
constexpr const char * PATCH_EMBED_B = "patch_embed.proj.bias";
constexpr const char * CLS_TOKEN = "cls_token";
constexpr const char * POS_EMBED = "pos_embed";
constexpr const char * FINAL_NORM_W = "norm.weight";
constexpr const char * FINAL_NORM_B = "norm.bias";
constexpr const char * HEAD_W = "head.weight";
constexpr const char * HEAD_B = "head.bias";

// Transformer block tensor name helpers.
inline std::string block_norm1_w(int i) {
    return "blocks." + std::to_string(i) + ".norm1.weight";
}
inline std::string block_norm1_b(int i) {
    return "blocks." + std::to_string(i) + ".norm1.bias";
}
inline std::string block_qkv_w(int i) {
    return "blocks." + std::to_string(i) + ".attn.qkv.weight";
}
inline std::string block_qkv_b(int i) {
    return "blocks." + std::to_string(i) + ".attn.qkv.bias";
}
inline std::string block_attn_proj_w(int i) {
    return "blocks." + std::to_string(i) + ".attn.proj.weight";
}
inline std::string block_attn_proj_b(int i) {
    return "blocks." + std::to_string(i) + ".attn.proj.bias";
}
inline std::string block_norm2_w(int i) {
    return "blocks." + std::to_string(i) + ".norm2.weight";
}
inline std::string block_norm2_b(int i) {
    return "blocks." + std::to_string(i) + ".norm2.bias";
}
inline std::string block_fc1_w(int i) {
    return "blocks." + std::to_string(i) + ".mlp.fc1.weight";
}
inline std::string block_fc1_b(int i) {
    return "blocks." + std::to_string(i) + ".mlp.fc1.bias";
}
inline std::string block_fc2_w(int i) {
    return "blocks." + std::to_string(i) + ".mlp.fc2.weight";
}
inline std::string block_fc2_b(int i) {
    return "blocks." + std::to_string(i) + ".mlp.fc2.bias";
}

// Predictor tensor name helpers.
inline std::string predictor_norm_w(int i) {
    return "score_predictor." + std::to_string(i) + ".in_conv.0.weight";
}
inline std::string predictor_norm_b(int i) {
    return "score_predictor." + std::to_string(i) + ".in_conv.0.bias";
}
inline std::string predictor_in_w(int i) {
    return "score_predictor." + std::to_string(i) + ".in_conv.1.weight";
}
inline std::string predictor_in_b(int i) {
    return "score_predictor." + std::to_string(i) + ".in_conv.1.bias";
}

inline std::string predictor_fc1_w(int i) {
    return "score_predictor." + std::to_string(i) + ".out_conv.0.weight";
}
inline std::string predictor_fc1_b(int i) {
    return "score_predictor." + std::to_string(i) + ".out_conv.0.bias";
}
inline std::string predictor_fc2_w(int i) {
    return "score_predictor." + std::to_string(i) + ".out_conv.2.weight";
}
inline std::string predictor_fc2_b(int i) {
    return "score_predictor." + std::to_string(i) + ".out_conv.2.bias";
}
inline std::string predictor_fc3_w(int i) {
    return "score_predictor." + std::to_string(i) + ".out_conv.4.weight";
}
inline std::string predictor_fc3_b(int i) {
    return "score_predictor." + std::to_string(i) + ".out_conv.4.bias";
}
} 

// Model loading
// Load DynamicViT from GGUF; returns nullptr on failure.
std::unique_ptr<model> load_model(const std::string & path);

// Model validation / debugging
// Validate that all required tensors exist and have expected shapes.
bool validate_model(const model & m);

// Print hyperparameters and tensor information.
void print_model_info(const model & m);

// Low-level graph helpers
ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias);
ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias, float eps);
ggml_tensor * gelu(ggml_context * ctx, ggml_tensor * x);

ggml_tensor * build_attention(ggml_context * ctx, const model & m, int block_index, ggml_tensor * x, int32_t n_tokens);
ggml_tensor * build_block(ggml_context * ctx, const model & m, int block_index, ggml_tensor * x, int32_t n_tokens);

// Predictor
ggml_tensor * build_predictor(ggml_context * ctx, const model & m, int predictor_index, ggml_tensor * x_patch, int32_t n_patch_tokens);
// Pruning
std::vector<int32_t> select_topk_tokens(const float * scores, int32_t n_tokens, int32_t n_keep);

// Postprocessing
int argmax(const std::vector<float> & logits);
std::vector<int> topk(const std::vector<float> & logits, int k);
bool contains(const std::vector<int> & v, int x);

// Patch convolution and flattening to [384, 196].
ggml_tensor * build_patch_embed(ggml_context * ctx, const model & m, ggml_tensor * image);

// Full inference
bool run_inference(model & m, const float * input_chw, inference_result & result);
std::vector<float> compact_tokens(const std::vector<float> & x, int32_t n_tokens, int32_t embed_dim, const std::vector<int32_t> & keep);
bool run_stage0(model & m, const float * input_chw, std::vector<float> & x_out, std::vector<float> & scores_out);
bool run_middle_stage(model & m, const std::vector<float> & input, int32_t n_tokens, int begin_block, int end_block, int predictor_index, std::vector<float> & x_out, std::vector<float> & scores_out);
bool run_final_stage(model & m, const std::vector<float> & input, int32_t n_tokens, std::vector<float> & logits);
} 

#endif 
