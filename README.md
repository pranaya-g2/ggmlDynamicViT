# DynamicViT with GGML

C++17 CPU inference and ImageNet evaluation for the 384-dimensional DynamicViT
model, using the GGML source in `ggml/` and an exported GGUF model.

## Requirements

- A C++17 compiler (Apple Clang on macOS, or GCC/Clang on Linux).
- CMake 3.16 or newer and a build tool such as Make.
- The bundled `ggml/` and `third_party/` directories.
- The exported model: `models/dynamic-vit_384_r0.7-f32.gguf`.

Python and PyTorch are not required to build or run this evaluator. The commands
below disable the Python parity tests and optional GPU backends for a CPU build.
They do not access or modify `dynamicViTorignal[donotedit]`.

## Compile

Run from the root of this repository:

```sh
cmake -S . -B build/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DGGML_BUILD_TESTS=OFF \
  -DGGML_BUILD_EXAMPLES=OFF \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_CCACHE=OFF

cmake --build build/cpu --target dynvit-eval -j 4
```

The executable is `build/cpu/dynvit-eval`. After editing C++ files, rerun the
second command to rebuild. CMake also writes `build/cpu/compile_commands.json`
for editors that need the compiler flags and include paths.

## Evaluate the image subset

The dataset must use numeric, **zero-based** ImageNet class directories:

```text
ILSVRC2012_img_val_subset/
  0/
    image.JPEG
  1/
    image.JPEG
  ...
  999/
    image.JPEG
```

Run all images in the subset:

```sh
build/cpu/dynvit-eval \
  models/dynamic-vit_384_r0.7-f32.gguf \
  ILSVRC2012_img_val_subset
```

The evaluator decodes and preprocesses each image, runs the model on the CPU,
and prints progress every 100 successfully evaluated images. Final output gives
the image count, top-1 accuracy, and top-5 accuracy. Failed image preprocessing
or inference calls are reported and skipped; check that the final count is
5,000 when evaluating the full supplied subset.

To also save the output in zsh or bash:

```sh
set -o pipefail
build/cpu/dynvit-eval \
  models/dynamic-vit_384_r0.7-f32.gguf \
  ILSVRC2012_img_val_subset 2>&1 | tee build/eval-results.txt
```

Output through `tee` can be buffered, so progress may appear in bursts. Run
without the pipe to see terminal progress sooner. Press Ctrl+C to stop a run;
an interrupted run does not produce final accuracy results.

## Scope

The current inference graph targets this specific model: 224×224 RGB inputs,
384 embedding dimensions, 12 transformer blocks, and three token-pruning stages.
Enabling a GPU build option alone does not switch the evaluator from its CPU
backend. For the separate PyTorch correctness tests, see [tests/README.md](tests/README.md).
