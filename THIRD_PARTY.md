# Third-party code and weights

- `qwen_spec/dflash_model.py`: the DFlash drafter model classes, vendored from
  [z-lab/dflash](https://github.com/z-lab/dflash) (MIT, Copyright (c) 2026 Z Lab) by way of mlx-dspark and trimmed
  to the classes used here; the DFlash 2 components (grouped convolution, candidate selector) follow the SGLang
  reference implementation (Apache-2.0). The license text is in the file header.
- `qwen_spec/qmm_small_m.py`, `qwen_spec/tree_attn.py`: the Metal kernels include, at JIT time, the
  `steel/gemm/nax.h` tensor-op header of the installed [MLX](https://github.com/ml-explore/mlx) package (MIT,
  Copyright Apple Inc.). No MLX source is copied into this repository.
- `qwen_spec/tree_verify.py`, `qwen_spec/mtp_drafter.py` build on the `qwen3_5` model code and cache classes of
  [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT).

Weights downloaded on first use (not part of this repository):

- Target: [lmstudio-community/Qwen3.8-27B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit),
  a 4-bit MLX conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).
- DFlash2 drafter: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (Apache-2.0).
- MTP head: [mlx-community/Qwen3.8-27B-MTP-bf16](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-bf16)
  (Apache-2.0, extracted from the Qwen release).
