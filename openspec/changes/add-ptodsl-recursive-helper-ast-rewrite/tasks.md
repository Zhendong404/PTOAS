## 1. Recursive Rewrite Infrastructure

- [x] 1.1 Add a tracing-time dispatch path that can detect source-backed plain
      Python helper callees and invoke them through recursive AST rewrite.
- [x] 1.2 Apply the same recursive-helper behavior to AST-rewrite-enabled
      `@pto.jit(entry=True)`, `@pto.jit(entry=False)`, and decorated
      `@pto.simd` / `@pto.cube` / `@pto.simt` roots.
- [x] 1.3 Add rewrite-session caching and in-progress tracking so repeated or
      cyclic helper calls do not re-rewrite endlessly.

## 2. Fallback And Cache Semantics

- [x] 2.1 Preserve the current fallback path for sourceless or unsupported
      plain Python callees.
- [x] 2.2 Extend AST-rewrite specialization keying so helper closure state that
      affects rewritten behavior invalidates stale compiled reuse.

## 3. Regression Coverage

- [x] 3.1 Add `ptodsl/tests/test_jit_compile.py` coverage for recursive rewrite
      through plain Python helper callees from entry kernels, kernel modules,
      and decorated PTODSL subkernels.
- [x] 3.2 Add fallback coverage showing that sourceless or unsupported plain
      helper callees still fail with the existing native-Python misuse
      diagnostic when they consume PTODSL runtime control-flow values.
- [x] 3.3 Add cache-regression coverage for helper closure state changes across
      recompilations.

## 4. Cv-Split Mix-Kernel Migration

- [x] 4.1 Update
      `ptodsl/examples/flash_attention/flash_attention_cv_split.py` to the
      single-entry mix-kernel shape with plain `cube_kernel` /
      `vector_kernel` helpers and no explicit authored helper `kernel_kind`.
- [x] 4.2 Update `ptodsl/tests/test_ptoas_frontend_verify.py` to validate the
      cv-split example as a mixed-section single-entry artifact instead of
      separate cube/vector `entry=False` helper children.
