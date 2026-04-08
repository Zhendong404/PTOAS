Qwen3 tilelet PTO kernels generated from `pypto-lib/examples/models/qwen3/qwen3_32b_decode_tilelet.py`.

Scope:
- direct `ptoas` compile-regression inputs
- A5-only kernels; `runop.sh` injects `--pto-arch a5 --pto-level=level3` for this directory unless the caller already overrides `PTOAS_FLAGS`

Notes:
- The current tilelet lowering emits 20 kernel fragments (`aiv`, `aic`, and mixed-kernel `.pto` files). This directory vendors those emitted `.pto` inputs directly, flattened into one sample directory for `runop.sh`.
- These files are regenerated from the tilelet example with `BATCH_TILE=16` / M=16 lowering.
- The directory is compile-regression focused; stale custom NPU-validation goldens for the old M=4 split are intentionally dropped here.
