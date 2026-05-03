## Validation Notes

Date: 2026-04-28

### Passed

- `cmake --build build --target ptoas -j2`
  - Result: passed, build tree was already up to date.
- Focused auto-vecscope tests:
  - `test/vpto/auto_vecscope_infer_simple.pto`
  - `test/vpto/auto_vecscope_infer_safe_scalar.pto`
  - `test/vpto/auto_vecscope_infer_boundary.pto`
  - `test/vpto/auto_vecscope_infer_nested_control_flow.pto`
  - `test/vpto/auto_vecscope_preserve_existing.pto`
  - `test/vpto/auto_vecscope_infer_escape_error.pto`
  - `test/vpto/auto_vecscope_infer_hivm_llvm_smoke.pto`
  - Result: 7 passed.
- TileLang DSL frontend tests with source checkout on `PYTHONPATH`:
  - `PYTHONPATH=$PWD/tilelang-dsl/python:$PWD/build/python python3 -m unittest discover -s tilelang-dsl/tests -p 'test_*.py'`
  - Result: passed. Confirms DSL semantic analysis no longer emits implicit
    `pto.vecscope` while explicit `pto.vecscope` / `pto.strict_vecscope`
    coverage remains valid.

### Full-suite blockers

- `lit -sv test/basic`
  - Result: 137 passed, 71 failed, 3 unresolved.
  - VecScope-related follow-up: update VPTO TileOp expansion FileCheck baselines
    that now see an inferred `pto.vecscope` before vector op sequences. Examples:
    `expand_tile_op_tilelang_trecip.pto`,
    `expand_tile_op_tilelang_texp.pto`,
    `expand_tile_op_tilelang_tabs.pto`,
    `tcolexpandadd_tilelang.pto`.
  - Existing-suite follow-up: several failures appear unrelated to this pass,
    including tests using unsupported `ptoas --pass-pipeline=...`, EmitC output
    formatting drift, and diagnostic wording drift.
- `lit -sv test/vpto`
  - Result: 8 passed, 257 unresolved.
  - Follow-up: `test/vpto/cases/**/kernel.pto` files are discovered by lit but
    most do not contain `RUN:` lines, so the full directory is not currently a
    clean lit target. Use the focused tests above or the VPTO validation scripts
    for these case directories until the lit configuration excludes or wraps
    no-RUN case files.

### Unsupported Edge Cases

- No new auto-vecscope inference edge case was found beyond the planned
  resultless-scope restriction: inferred clusters are rejected when
  `!pto.vreg`, `!pto.mask`, or `!pto.align` results have users outside the
  inferred cluster.
- Closing issue #122 should be blocked on either updating the affected VPTO
  TileOp expansion FileCheck baselines or explicitly tracking that baseline
  refresh separately.
