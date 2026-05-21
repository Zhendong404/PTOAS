# JIT examples

Python `@pto.jit` kernels with compile-only and end-to-end launch smoke tests.

## Prerequisites

- ptoas + ptodsl installed per [ptodsl README](../README.md) (`quick_install.sh`, `pip install -e .`)
- CANN 9.0+ with `ASCEND_HOME_PATH` set
- For end-to-end launch: `torch`, `torch_npu`, `numpy`; `bisheng` on PATH

## Environment (every shell)

```bash
cd $PTOAS_REPO_ROOT          # e.g. /workdir/ptoas_a5
source set_ptoas_env.sh
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
```

For CPU simulation (msprof), also:

```bash
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH}"
ulimit -n 65535
```

## 1. Compile-only: DSL → MLIR (`tadd_builder.py`)

Verifies Python `@pto.jit` emits the same IR as the TileLang ST `tadd.pto` testcase.

```bash
cd ptodsl/examples/jit
python3 tadd_builder.py
```

Expected: MLIR module text printed to stdout containing `@TADD_f32_16x64` and `@TADD_f32_32x32`.

Optional — run through the ptoas frontend:

```bash
python3 tadd_builder.py > /tmp/tadd_dsl.mlir
ptoas --emit-pto-ir /tmp/tadd_dsl.mlir -o - | head
```

## 2. End-to-end: DSL → IR → binary → launch → accuracy (`tadd_launch.py`)

Self-contained smoke test with inline golden generation and accuracy checks. Runs under the msprof CPU simulator — no physical NPU required.

```bash
cd ptodsl/examples/jit
msprof op simulator --soc-version=Ascend950PR_9599 \
  --output=msprof_res/tadd \
  python3 tadd_launch.py
```

Expected output:

```
PASS f32_16x64
PASS f32_32x32
All cases passed.
```

Direct run on a real NPU (omit the msprof wrapper when hardware is available):

```bash
cd ptodsl/examples/jit
python3 tadd_launch.py
```

## Artifacts (gitignored)

- `~/.cache/ptodsl/` — JIT-compiled kernel `.so` cache (override with `PTODSL_CACHE_DIR`)
- `msprof_res/` — msprof simulator trace output
