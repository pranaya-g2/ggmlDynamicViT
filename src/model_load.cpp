#include "dynvit-impl.h"
#include "../ggml/include/gguf.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dynvit {

static int64_t require_key(const gguf_context * gguf, const char * key ) {
    const int64_t id = gguf_find_key(gguf, key);

    if (id < 0) {
        throw std::runtime_error(std::string("Missing GGUF key: ") + key );
    }

    return id;
}

static ggml_tensor * require_tensor(ggml_context * ctx, const std::string & name ) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());

    if (t == nullptr) {
        throw std::runtime_error("Missing tensor: " + name );
    }

    return t;
}

static void load_hparams(const gguf_context * gguf, hparams & hp ) {
    {
        const int64_t id = require_key(gguf, gguf_key::ARCHITECTURE);
        const char * arch = gguf_get_val_str(gguf, id);

        if (std::strcmp(arch, "dynamicvit") != 0) {
            throw std::runtime_error(std::string("Expected architecture 'dynamicvit', got '") + arch + "'" );
        }
    }

    hp.image_size = gguf_get_val_u32(gguf, require_key(gguf, gguf_key::IMAGE_SIZE) );
    hp.patch_size = gguf_get_val_u32(gguf, require_key(gguf, gguf_key::PATCH_SIZE) );
    hp.in_channels = gguf_get_val_u32(gguf, require_key(gguf, gguf_key::IN_CHANNELS) );
    hp.embed_dim = gguf_get_val_u32(gguf, require_key(gguf, gguf_key::EMBED_DIM) );
    hp.num_heads = gguf_get_val_u32(gguf, require_key(gguf, gguf_key::NUM_HEADS) );
    hp.num_blocks = gguf_get_val_u32(gguf, require_key(gguf, gguf_key::NUM_BLOCKS) );
    hp.mlp_dim = gguf_get_val_u32(gguf, require_key(gguf, gguf_key::MLP_DIM) );
    hp.num_classes = gguf_get_val_u32(gguf, require_key(gguf, gguf_key::NUM_CLASSES) );
    hp.norm_eps = gguf_get_val_f32(gguf, require_key(gguf, gguf_key::NORM_EPS) );
    hp.head_dim = hp.embed_dim / hp.num_heads;
    const int patches_per_side = hp.image_size / hp.patch_size;
    hp.num_patches = patches_per_side * patches_per_side;

    // Pruning layers.
    {
        const int64_t id = require_key(gguf, gguf_key::PRUNE_LAYERS );
        const size_t n = gguf_get_arr_n(gguf, id);

        if (n != DV_NUM_PREDICTORS) {
            throw std::runtime_error("dynamicvit.prune_layers must contain exactly 3 entries" );
        }

        const uint32_t * values = static_cast<const uint32_t *>(gguf_get_arr_data(gguf, id) );

        for (int i = 0; i < DV_NUM_PREDICTORS; ++i) {
            hp.prune_layers[i] = static_cast<int32_t>(values[i]);
        }
    }

    // Keep ratios.
    {
        const int64_t id = require_key(gguf, gguf_key::KEEP_RATIOS );
        const size_t n = gguf_get_arr_n(gguf, id);

        if (n != DV_NUM_PREDICTORS) {
            throw std::runtime_error("dynamicvit.keep_ratios must contain exactly 3 entries" );
        }

        const float * values = static_cast<const float *>(gguf_get_arr_data(gguf, id) );

        for (int i = 0; i < DV_NUM_PREDICTORS; ++i) {
            hp.keep_ratios[i] = values[i];
        }
    }
}

// Bind model fields to tensor metadata created by GGUF.

static void load_tensor_pointers(model & m ) {
    ggml_context * ctx = m.ctx;
    m.patch_embed_w = require_tensor(ctx, tensor_name::PATCH_EMBED_W );
    m.patch_embed_b = require_tensor(ctx, tensor_name::PATCH_EMBED_B );
    m.cls_token = require_tensor(ctx, tensor_name::CLS_TOKEN );
    m.pos_embed = require_tensor(ctx, tensor_name::POS_EMBED );

    for (int i = 0; i < m.hp.num_blocks; ++i) {
        auto & b = m.blocks[i];
        b.norm1_w = require_tensor(ctx, tensor_name::block_norm1_w(i) );
        b.norm1_b = require_tensor(ctx, tensor_name::block_norm1_b(i) );
        b.qkv_w = require_tensor(ctx, tensor_name::block_qkv_w(i) );
        b.qkv_b = require_tensor(ctx, tensor_name::block_qkv_b(i) );
        b.attn_proj_w = require_tensor(ctx, tensor_name::block_attn_proj_w(i) );
        b.attn_proj_b = require_tensor(ctx, tensor_name::block_attn_proj_b(i) );
        b.norm2_w = require_tensor(ctx, tensor_name::block_norm2_w(i) );
        b.norm2_b = require_tensor(ctx, tensor_name::block_norm2_b(i) );
        b.mlp_fc1_w = require_tensor(ctx, tensor_name::block_fc1_w(i) );
        b.mlp_fc1_b = require_tensor(ctx, tensor_name::block_fc1_b(i) );
        b.mlp_fc2_w = require_tensor(ctx, tensor_name::block_fc2_w(i) );
        b.mlp_fc2_b = require_tensor(ctx, tensor_name::block_fc2_b(i) );
    }

    for (int i = 0; i < DV_NUM_PREDICTORS; ++i) {
        auto & p = m.predictors[i];
        p.norm_w = require_tensor(ctx, tensor_name::predictor_norm_w(i) );
        p.norm_b = require_tensor(ctx, tensor_name::predictor_norm_b(i) );
        p.in_w = require_tensor(ctx, tensor_name::predictor_in_w(i) );
        p.in_b = require_tensor(ctx, tensor_name::predictor_in_b(i) );
        p.fc1_w = require_tensor(ctx, tensor_name::predictor_fc1_w(i) );
        p.fc1_b = require_tensor(ctx, tensor_name::predictor_fc1_b(i) );
        p.fc2_w = require_tensor(ctx, tensor_name::predictor_fc2_w(i) );
        p.fc2_b = require_tensor(ctx, tensor_name::predictor_fc2_b(i) );
        p.fc3_w = require_tensor(ctx, tensor_name::predictor_fc3_w(i) );
        p.fc3_b = require_tensor(ctx, tensor_name::predictor_fc3_b(i) );
    }

    m.final_norm_w = require_tensor(ctx, tensor_name::FINAL_NORM_W );
    m.final_norm_b = require_tensor(ctx, tensor_name::FINAL_NORM_B );
    m.head_w = require_tensor(ctx, tensor_name::HEAD_W );
    m.head_b = require_tensor(ctx, tensor_name::HEAD_B );
}

static bool check_1d(const ggml_tensor * t, int64_t n0, const char * name ) {
    if (t == nullptr) {
        std::fprintf(stderr, "Tensor %s is null\n", name );

        return false;
    }

    if (t->ne[0] != n0) {
        std::fprintf(stderr, "%s: expected ne[0]=%lld, got %lld\n", name, static_cast<long long>(n0), static_cast<long long>(t->ne[0]) );

        return false;
    }

    return true;
}

static bool check_2d(const ggml_tensor * t, int64_t n0, int64_t n1, const char * name ) {
    if (t == nullptr) {
        std::fprintf(stderr, "Tensor %s is null\n", name );

        return false;
    }

    if (t->ne[0] != n0 || t->ne[1] != n1 ) {
        std::fprintf(stderr, "%s: expected [%lld, %lld], got [%lld, %lld]\n", name, static_cast<long long>(n0), static_cast<long long>(n1), static_cast<long long>(t->ne[0]), static_cast<long long>(t->ne[1]) );

        return false;
    }

    return true;
}

bool validate_model(const model & m ) {
    bool ok = true;

    if (m.hp.image_size != 224) {
        std::fprintf(stderr, "Expected image_size=224\n" );
        ok = false;
    }

    if (m.hp.patch_size != 16) {
        std::fprintf(stderr, "Expected patch_size=16\n" );
        ok = false;
    }

    if (m.hp.embed_dim != 384) {
        std::fprintf(stderr, "Expected embed_dim=384\n" );
        ok = false;
    }

    if (m.hp.num_heads != 6) {
        std::fprintf(stderr, "Expected num_heads=6\n" );
        ok = false;
    }

    if (m.hp.num_blocks != 12) {
        std::fprintf(stderr, "Expected num_blocks=12\n" );
        ok = false;
    }

    if (m.hp.mlp_dim != 1536) {
        std::fprintf(stderr, "Expected mlp_dim=1536\n" );
        ok = false;
    }

    if (m.hp.num_classes != 1000) {
        std::fprintf(stderr, "Expected num_classes=1000\n" );
        ok = false;
    }

    ok &= check_1d(m.patch_embed_b, m.hp.embed_dim, "patch_embed.bias" );
    ok &= check_1d(m.final_norm_w, m.hp.embed_dim, "norm.weight" );
    ok &= check_1d(m.final_norm_b, m.hp.embed_dim, "norm.bias" );
    ok &= check_1d(m.head_b, m.hp.num_classes, "head.bias" );

    // GGML linear weights use ne[0] = in_features and ne[1] = out_features.

    for (int i = 0; i < m.hp.num_blocks; ++i) {
        const auto & b = m.blocks[i];
        const int E = m.hp.embed_dim;
        const int M = m.hp.mlp_dim;
        ok &= check_1d(b.norm1_w, E, "block.norm1.weight" );
        ok &= check_1d(b.norm1_b, E, "block.norm1.bias" );
        ok &= check_2d(b.qkv_w, E, 3 * E, "block.attn.qkv.weight" );
        ok &= check_1d(b.qkv_b, 3 * E, "block.attn.qkv.bias" );
        ok &= check_2d(b.attn_proj_w, E, E, "block.attn.proj.weight" );
        ok &= check_1d(b.attn_proj_b, E, "block.attn.proj.bias" );
        ok &= check_1d(b.norm2_w, E, "block.norm2.weight" );
        ok &= check_1d(b.norm2_b, E, "block.norm2.bias" );
        ok &= check_2d(b.mlp_fc1_w, E, M, "block.mlp.fc1.weight" );
        ok &= check_1d(b.mlp_fc1_b, M, "block.mlp.fc1.bias" );
        ok &= check_2d(b.mlp_fc2_w, M, E, "block.mlp.fc2.weight" );
        ok &= check_1d(b.mlp_fc2_b, E, "block.mlp.fc2.bias" );
    }

    ok &= check_2d(m.head_w, m.hp.embed_dim, m.hp.num_classes, "head.weight" );

    return ok;
}

void print_model_info(const model & m ) {
    std::printf("DynamicViT model\n" "----------------\n" );
    std::printf("image size      : %d\n", m.hp.image_size );
    std::printf("patch size      : %d\n", m.hp.patch_size );
    std::printf("patches         : %d\n", m.hp.num_patches );
    std::printf("embedding dim   : %d\n", m.hp.embed_dim );
    std::printf("heads           : %d\n", m.hp.num_heads );
    std::printf("head dim        : %d\n", m.hp.head_dim );
    std::printf("blocks          : %d\n", m.hp.num_blocks );
    std::printf("MLP dim         : %d\n", m.hp.mlp_dim );
    std::printf("classes         : %d\n", m.hp.num_classes );
    std::printf("LayerNorm eps   : %.8g\n", m.hp.norm_eps );
    std::printf("prune layers    : %d %d %d\n", m.hp.prune_layers[0], m.hp.prune_layers[1], m.hp.prune_layers[2] );
    std::printf("keep ratios     : %.3f %.3f %.3f\n", m.hp.keep_ratios[0], m.hp.keep_ratios[1], m.hp.keep_ratios[2] );
}

std::unique_ptr<model> load_model(const std::string & path ) {
    auto m = std::make_unique<model>();

    // Parse GGUF tensor metadata without allocating tensor data.
    ggml_context * ctx = nullptr;
    gguf_init_params params = {
        true, // no_alloc
        &ctx, // ctx
    };
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params );

    if (gguf == nullptr) {
        throw std::runtime_error("Failed to open GGUF model: " + path );
    }

    if (ctx == nullptr) {
        gguf_free(gguf);
        throw std::runtime_error("GGUF loaded but GGML context was null" );
    }

    m->ctx = ctx;
    load_hparams(gguf, m->hp );
    load_tensor_pointers(*m);

    // Validate tensor shapes before allocating model data.

    if (!validate_model(*m)) {
        gguf_free(gguf);
        throw std::runtime_error("DynamicViT model validation failed" );
    }

    m->backend = ggml_backend_cpu_init();

    if (m->backend == nullptr) {
        gguf_free(gguf);
        throw std::runtime_error("Failed to initialize GGML CPU backend" );
    }

    // Allocate storage for all tensors using the CPU backend.
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(m->backend );
    m->buffer = ggml_backend_alloc_ctx_tensors_from_buft(m->ctx, buft );

    if (m->buffer == nullptr) {
        gguf_free(gguf);
        throw std::runtime_error("Failed to allocate backend model buffer" );
    }

    // Read each tensor from the GGUF data section into backend storage.
    std::FILE * fp = std::fopen(path.c_str(), "rb" );

    if (fp == nullptr) {
        gguf_free(gguf);
        throw std::runtime_error("Could not reopen GGUF file for tensor loading" );
    }

    const size_t data_offset = gguf_get_data_offset(gguf);

    for (ggml_tensor * t = ggml_get_first_tensor(m->ctx); t != nullptr; t = ggml_get_next_tensor(m->ctx, t) ) {
        const int64_t tensor_index = gguf_find_tensor(gguf, t->name );

        if (tensor_index < 0) {
            std::fclose(fp);
            gguf_free(gguf);
            throw std::runtime_error(std::string("Tensor not found in GGUF index: ") + t->name );
        }

        const size_t tensor_offset = gguf_get_tensor_offset(gguf, tensor_index );
        const size_t tensor_size = ggml_nbytes(t);
        std::vector<uint8_t> temp(tensor_size );
        const size_t absolute_offset = data_offset + tensor_offset;

        if (std::fseek(fp, static_cast<long>(absolute_offset), SEEK_SET ) != 0 ) {
            std::fclose(fp);
            gguf_free(gguf);
            throw std::runtime_error(std::string("Failed seeking to tensor: ") + t->name );
        }

        const size_t nread = std::fread(temp.data(), 1, tensor_size, fp );

        if (nread != tensor_size) {
            std::fclose(fp);
            gguf_free(gguf);
            throw std::runtime_error(std::string("Failed reading tensor: ") + t->name );
        }

        ggml_backend_tensor_set(t, temp.data(), 0, tensor_size );
    }

    std::fclose(fp);

    // Release GGUF metadata; the tensor context remains alive in m->ctx.
    gguf_free(gguf);

    return m;
}

}
