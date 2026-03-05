# PTOAS OP-Lib IR Spec (V1)

## 1. Goal

This document defines the OP-Lib template contract used by PTOAS tile-fusion V1.

V1 focuses on binary floating elementwise ops in fused chains:

- `tadd`
- `tsub`
- `tmul`
- `tdiv`

The key design is decoupling function naming from `dtype` and `shape`:

- function symbols express only the OP identity
- shape uses dynamic memref ranks (`?x?`)
- seed template dtype is `f32`
- PTOAS auto-instantiates `f16` / `f32` instances on demand

## 2. Required Function Attributes

Each template function must carry these attributes:

- `pto.oplib.op` : `"tadd" | "tsub" | "tmul" | "tdiv"`
- `pto.oplib.kind` : `"binary_elementwise_template"`
- `pto.oplib.rank` : `2 : i64`
- `pto.oplib.seed_dtype` : `"f32"`

If any required attribute is missing or invalid, PTOAS fails hard.

## 3. Naming Convention

Template symbol names are fixed-form for readability, but matching is attribute-driven.

Recommended names:

- `@__pto_oplib_tadd_template`
- `@__pto_oplib_tsub_template`
- `@__pto_oplib_tmul_template`
- `@__pto_oplib_tdiv_template`

Notes:

- PTOAS does not rely on suffixes like `__f32__32x32` anymore.
- `dtype` / `shape` are no longer encoded in symbol names.

## 4. Signature Contract

Template function signature must be:

```mlir
(memref<?x?xT, #pto.address_space<vec>>,
 memref<?x?xT, #pto.address_space<vec>>,
 memref<?x?xT, #pto.address_space<vec>>) -> ()
```

V1 seed template requires `T=f32`.

## 5. Body Constraints (Hard Rules)

Allowed dialects/op families (V1):

- `func`
- `memref`
- `scf`
- `arith`

Required structure:

- 2-level loop nest (`scf.for` outer/inner)
- per-element compute pattern: load/load/compute/store

Forbidden in template body:

- `memref.alloc` / `memref.dealloc`
- `func.call`
- `scf.if` / `scf.while`
- side-effectful ops unrelated to elementwise compute

## 6. OP-to-Arith Mapping (V1)

- `tadd` -> `arith.addf`
- `tsub` -> `arith.subf`
- `tmul` -> `arith.mulf`
- `tdiv` -> `arith.divf`

## 7. PTOAS Runtime Behavior

When fusion is enabled:

1. PTOAS scans `--op-lib-dir` and imports template functions.
2. PTOAS matches template by function attributes.
3. PTOAS instantiates target dtype function per `(op, dtype)` and caches it.
4. PTOAS materializes fused group functions by calling instantiated OP-Lib symbols.
5. In V1, instantiated symbols are emitted as declarations (external call boundary);
   template bodies are validated for contract conformance but not inlined into fused functions.

Failure policy:

- missing template / invalid attrs / unsupported dtype => compile error
- no silent fallback

## 8. Example Template (`tmul`)

```mlir
func.func @__pto_oplib_tmul_template(
    %src0: memref<?x?xf32, #pto.address_space<vec>>,
    %src1: memref<?x?xf32, #pto.address_space<vec>>,
    %dst:  memref<?x?xf32, #pto.address_space<vec>>)
    attributes {
      pto.oplib.op = "tmul",
      pto.oplib.kind = "binary_elementwise_template",
      pto.oplib.rank = 2 : i64,
      pto.oplib.seed_dtype = "f32"
    } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = memref.dim %dst, %c0 : memref<?x?xf32, #pto.address_space<vec>>
  %n = memref.dim %dst, %c1 : memref<?x?xf32, #pto.address_space<vec>>

  scf.for %i = %c0 to %m step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %a = memref.load %src0[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
      %b = memref.load %src1[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
      %v = arith.mulf %a, %b : f32
      memref.store %v, %dst[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
    }
  }
  return
}
```
