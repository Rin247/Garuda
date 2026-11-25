# Garuda — Accelerated LLM Inference Engine & APEX Quantization Fork

Garuda is an optimized fork of `llama.cpp` featuring high-precision APEX quantization profiles, advanced memory layout optimizations, and direct hardware bindings for CPU, CUDA, Vulkan, and Metal backends. Garuda serves as the underlying C++ inference foundation for offline AI engines and desktop workbenches.

---

## 🚀 Key Features & Module Breakdown

### 🔬 APEX Quantization Engine (`tools/quantize/apex`)
- **Current**: APEX per-tensor quantization profiles, specialized mixed-precision GGUF quantization (e.g. IQ2_XXS up to Q8_0), and outlier-preserving weight scaling.
- **Planned / Roadmap**: Automated layer-by-layer perplexity loss optimizer, sub-byte FP2/FP3 quantization, and KV-cache compression.

### ⚡ Accelerated CPU & GPU Backends (`ggml`)
- **Current**: High-performance FlashAttention, AVX2/AVX-512/AMX vectorization, CUDA cuBLAS acceleration, and Vulkan compute pipelines.
- **Planned / Roadmap**: Multi-GPU tensor parallelism scaling improvements, dynamic context caching, and native NPU execution backends.

### 🔄 Model Conversion & GGUF Forge (`gguf-py`)
- **Current**: Direct Hugging Face PyTorch/Safetensors to GGUF model converter, vocabulary inspector, and tensor metadata editor.
- **Planned / Roadmap**: Instant GGUF tensor de-quantization preview, model merging/stitching tool, and LoRA weight baker.

---

## 🏗️ Architecture & Technology Stack

- **Core Engine**: C++20, C11 (`ggml`), CUDA, Vulkan, Metal.
- **Tools**: Python 3.15 (`gguf-py` conversion toolkit).
- **Licensing**: MIT License (upstream `llama.cpp` compatibility retained).

---

## 🛠️ Building & Running

### Prerequisites
- **Visual Studio 2022** (C++ MSVC compiler) or **GCC / Clang**
- **CMake 3.25+**
- **CUDA Toolkit** (Optional for GPU acceleration)

### Build Commands

```bat
:: Build Release binaries (llama-cli, llama-quantize, llama-server)
build.bat Release

:: Run llama-cli
run.bat
```

---

## 📜 License & Acknowledgements

Garuda is licensed under the **MIT License**. Upstream copyright belongs to `ggml-org / llama.cpp` contributors. See [`LICENSE`](LICENSE) for details.
