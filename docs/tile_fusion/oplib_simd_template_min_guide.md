# OP-Lib `pto.simd` Authoring Quickstart

- 状态：Draft v2.0
- 适用范围：Level-3 OP-Lib `pto.simd` 模板开发
- 权威规范：[oplib_ir_spec.md](./oplib_ir_spec.md)

本文不是第二份 spec，只是上手指南。模板契约、命名规则、family 签名、错误码与校验规则都以 [oplib_ir_spec.md](./oplib_ir_spec.md) 为准。

## 1. 先记住五条硬约束

1. 对外 ABI 不变：函数签名继续由 `pto.oplib.kind` 决定，tile-like 参数仍是 `!pto.tile_buf`
2. 模板体中的计算 / 访存语义统一写成 `pto.simd.*`
3. 允许的非 `pto.simd` 辅助 IR 只保留 `func`、`scf`、`arith.constant`、`arith.index_cast` 和必要 `memref` 视图辅助
4. 新模板不再把 `vector.*` / `arith.*` / `math.*` 当作规范 authoring 接口
5. scalar 变体不再用 `...s` 后缀，而是复用同名 `pto.simd` op 并通过操作数类型区分

## 2. 最小上手流程

1. 先确定 template `kind` 与签名类别
2. 写齐 `pto.oplib.*` 元数据，特别是 `argN.*`
3. 声明 `pto.simd.level` 与 `pto.simd.lanes`
4. 用 `pto.simd.tile_to_memref` 暴露 memref bridge
5. 用 `pto.simd.load/store` 或 structured `pto.simd.*` op 写模板体语义
6. 最后补 A5 专用属性：`pto.simd.vld_dist`、`pto.simd.vst_dist`、`pto.simd.exec_mode`

## 3. 骨架 A：`pto.simd.add`

这是一个最小 lane-level binary arithmetic 模板，适合 `l3_float_binary_elementwise_template`。

```mlir
func.func private @__pto_oplib_variant_simd_add(
    %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %dst:  !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    attributes {
      pto.oplib.kind = "l3_float_binary_elementwise_template",
      pto.oplib.entry_role = "variant",
      pto.oplib.op = "tadd",
      pto.oplib.variant_id = "simd_add_rm_f32",
      pto.oplib.match.dtype = "f32",
      pto.oplib.match.arg0.rows = -1 : i64,
      pto.oplib.match.arg0.cols = -1 : i64,
      pto.oplib.match.arg0.blayout = "row_major",
      pto.oplib.match.arg0.slayout = "any",
      pto.oplib.match.arg0.fractal = -1 : i64,
      pto.oplib.match.arg1.rows = -1 : i64,
      pto.oplib.match.arg1.cols = -1 : i64,
      pto.oplib.match.arg1.blayout = "row_major",
      pto.oplib.match.arg1.slayout = "any",
      pto.oplib.match.arg1.fractal = -1 : i64,
      pto.oplib.match.arg2.rows = -1 : i64,
      pto.oplib.match.arg2.cols = -1 : i64,
      pto.oplib.match.arg2.blayout = "row_major",
      pto.oplib.match.arg2.slayout = "any",
      pto.oplib.match.arg2.fractal = -1 : i64,
      pto.oplib.cost = 10 : i64,
      pto.oplib.priority = 0 : i64,
      pto.simd.level = "binary_ewise_v1",
      pto.simd.lanes = 64 : i64
    } {
  %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>
  %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>
  %md = pto.simd.tile_to_memref %dst  : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>

  %flat0 = memref.reinterpret_cast %m0 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
  %flat1 = memref.reinterpret_cast %m1 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
  %flatd = memref.reinterpret_cast %md to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>

  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index

  scf.for %i = %c0 to %c1024 step %c64 {
    %mask = pto.simd.predicate %c64 : index -> vector<64xi1>
    %a = pto.simd.load %flat0, %i, %mask {pto.simd.vld_dist = "NORM"} : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> vector<64xf32>
    %b = pto.simd.load %flat1, %i, %mask {pto.simd.vld_dist = "NORM"} : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> vector<64xf32>
    %c = pto.simd.add %a, %b {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>, vector<64xf32> -> vector<64xf32>
    pto.simd.store %c, %flatd, %i, %mask {pto.simd.vst_dist = "DIST_NORM"} : vector<64xf32>, memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1>
  }
  return
}
```

如果你在 `seed` 里复用这类骨架，唯一可改写的 core slot 要标在 `pto.simd.add/sub/mul/div/max/min` 上，而不是历史 `arith.*f` 上。

## 4. 骨架 B：`pto.simd.row_sum`

这是一个最小 structured memory-level 模板，适合 `l3_reduce_row_template`。

```mlir
func.func private @__pto_oplib_variant_row_sum(
    %src: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    attributes {
      pto.oplib.kind = "l3_reduce_row_template",
      pto.oplib.entry_role = "variant",
      pto.oplib.op = "trowsum",
      pto.oplib.variant_id = "simd_row_sum_f32",
      pto.oplib.match.dtype = "f32",
      pto.oplib.match.arg0.rows = -1 : i64,
      pto.oplib.match.arg0.cols = -1 : i64,
      pto.oplib.match.arg0.blayout = "row_major",
      pto.oplib.match.arg0.slayout = "any",
      pto.oplib.match.arg0.fractal = -1 : i64,
      pto.oplib.match.arg1.rows = -1 : i64,
      pto.oplib.match.arg1.cols = -1 : i64,
      pto.oplib.match.arg1.blayout = "row_major",
      pto.oplib.match.arg1.slayout = "any",
      pto.oplib.match.arg1.fractal = -1 : i64,
      pto.oplib.cost = 10 : i64,
      pto.oplib.priority = 0 : i64,
      pto.simd.level = "row_reduce_v1",
      pto.simd.lanes = 64 : i64
    } {
  %ms = pto.simd.tile_to_memref %src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, #pto.address_space<vec>>
  %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x1xf32, #pto.address_space<vec>>
  pto.simd.row_sum %ms, %md : memref<32x32xf32, #pto.address_space<vec>>, memref<32x1xf32, #pto.address_space<vec>>
  return
}
```

这类 structured op 适合直接表达 reduction / broadcast family 的规范语义；当 family ABI 需要 `tmp` 时，要按 spec 中约定把 `tmp` 显式放进模板签名和模板体调用。

## 5. Legacy 名称到新 `pto.simd` 的最小对照

| Legacy mixed IR | 新 authoring 名称 |
|-----------------|-------------------|
| `vector.load` | `pto.simd.load` |
| `vector.store` | `pto.simd.store` |
| `arith.addf` / `arith.subf` / `arith.mulf` / `arith.divf` | `pto.simd.add` / `sub` / `mul` / `div` |
| `arith.maximumf` / `arith.minimumf` | `pto.simd.max` / `min` |
| `math.exp` / `math.log` / `math.sqrt` / `math.rsqrt` | `pto.simd.exp` / `log` / `sqrt` / `rsqrt` |
| `arith.cmpf` 风格比较 | `pto.simd.cmp` |
| `vector.select` 风格选择 | `pto.simd.select` |

## 6. 自检清单

1. `kind` 与函数签名匹配
2. 所有 tile-like 参数都写齐 `pto.oplib.match.argN.*`
3. 新模板体里没有把 `vector.*` / `arith.*` / `math.*` 当主体计算语义使用
4. lane-level op 的类型宽度与 `pto.simd.lanes` 一致
5. A5 路径上 `load/store` 和计算 op 已补齐 `vld_dist` / `vst_dist` / `exec_mode`
6. `seed` 模板若声明 core slot，只出现一个合法 `pto.simd.add/sub/mul/div/max/min`
7. reduction / broadcast family 若需要 `tmp` 或 `is_binary`，已经与 family 契约对齐
