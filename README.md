# DynamicViT with GGML

C++ CPU inference for DynamicViT-DeiT-S/0.7 using GGML.

The implementation includes GGUF model loading, ImageNet preprocessing,
Vision Transformer blocks, DynamicViT token pruning, and top-1/top-5
ImageNet evaluation. The implementation is validated against the official python implementation by Rao et. al (authors of DViT), and results matched. I trained on a subset of the validation set: https://github.com/ndb796/Small-ImageNet-Validation-Dataset-1000-Classes which has 5000 images. On mac, I ran my ggml version on 2500 images (273 seconds on a sample run) and the classification accurarcy matched up to the Pytorch implementation of the same dataset, though the pytorch version was faster (50 seconds). This is expected since the ggml overhead of my implementation. It creates and frees a graph, scheduler, and working buffers for each of the four inference stages. That adds allocation and scheduling overhead. Another possible reason would be that token features and predictor scores are copied into C++ vectors, pruned, and uploaded into the next graph. I also ran it on only 2 threads and disabled the BLAS ggml backend. I am planning on further optimizing as per the TODO at the end. 

## Setup

Clone GGML into the `ggml/` directory. Follow ggml repo's instructions for that.

Place `stb_image.h` in:

```text
third_party/stb_image.h
```

The expected GGUF model path is:

```text
models/dynamic-vit_384_r0.7-f32.gguf
```

The model file is not included in this repository. Please refer to the DViT paper and their repositary. 

## Build

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DGGML_BUILD_TESTS=OFF \
  -DGGML_BUILD_EXAMPLES=OFF \
  -DGGML_METAL=OFF

cmake --build build --target dynvit-eval -j
```

The evaluator executable will be:

```text
build/dynvit-eval
```

## Dataset Layout

The ImageNet validation subset should use numeric class directories:

```text
ILSVRC2012_img_val_subset/
├── 1/
├── 2/
├── ...
└── 1000/
```

## Run Evaluation

```sh
build/dynvit-eval \
  models/dynamic-vit_384_r0.7-f32.gguf \
  ILSVRC2012_img_val_subset
```

The evaluator prints running top-1 and top-5 accuracy every 100 images and
reports final accuracy after the dataset has been processed.

## Model

This implementation targets DynamicViT-DeiT-S/0.7:

- Input: 224 × 224 RGB
- Embedding dimension: 384
- Transformer blocks: 12
- Attention heads: 6
- MLP dimension: 1536
- Dynamic token pruning stages: 3
- ImageNet classes: 1000

Inference uses the DynamicViT token schedule as per the official paper:

```text
197 -> 138 -> 97 -> 68 tokens
```

All model computation is executed through GGML on the CPU.

## TODO

The current implementation establishes a functional GGML baseline that reproduces the classification accuracy of the original PyTorch implementation. The repo is actively changing to optimize the inference path for lower latency and better CPU efficiency:

- [ ] Profile DynamicViT inference to identify CPU bottlenecks
- [ ] Implement fused kernels for DynamicViT-specific compute patterns
- [ ] Optimize token compaction/gather and pruning overhead
