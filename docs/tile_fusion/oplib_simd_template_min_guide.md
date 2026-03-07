# OP-Lib 开发者最小指南：编写 `pto.simd.*` 模板（V1）

- 状态：Draft v1.0
- 适用范围：Level-3 OP-Lib Binary Element-Wise（`tadd/tsub/tmul/tdiv/tmax/tmin`）
- 目标读者：编写/维护 OP-Lib 模板的开发者

## 1. 先记住三条硬约束

1. 对外 ABI 不变：函数签名必须是 `(!pto.tile_buf, !pto.tile_buf, !pto.tile_buf) -> ()`。
2. 模板函数体必须有 `pto.simd.*` 语义骨架，禁止空 body 回退。
3. V1 只支持 `f16/f32 + row_major`，不支持其他 dtype/layout。

## 2. 最小上手流程

1. 先决定模板角色：`variant` 或 `seed`。
2. 写齐 `pto.oplib.*` 匹配元数据。
3. 写齐 `pto.simd.level` 与 `pto.simd.lanes`。
4. 在函数体中按 `load -> core -> store` 组织逻辑。
5. 对 `seed`，只保留一个 `pto.simd.core_slot = "binary_ewise_core"` 的核心算术 op。
6. 用 `ptoas --op-lib-dir=...` 跑一轮导入和实例化自检。

## 3. 角色差异（`variant` vs `seed`）

| 维度 | `variant` | `seed` |
| --- | --- | --- |
| `pto.oplib.entry_role` | `"variant"` | `"seed"` |
| 需要的操作元数据 | `pto.oplib.op`, `pto.oplib.variant_id`, `pto.oplib.match.dtype` | `pto.oplib.seed_id`, `pto.oplib.seed_dtype`, `pto.oplib.seed.support_dtypes`, `pto.oplib.seed.support_ops`, `pto.oplib.seed.core_slot` |
| 核心算术 op 的改写 | 不改写，直接使用模板体算术语义 | 实例化时仅改写 core slot 对应算术 op |
| 推荐使用场景 | 某个 op 的专门优化实现 | 一套骨架覆盖多个 binary op |

## 4. 最小模板骨架（可直接改）

```mlir
func.func private @__pto_oplib_seed_vec_bin_core(
    %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %dst:  !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    attributes {
      pto.oplib.kind = "l3_binary_elementwise_template",
      pto.oplib.entry_role = "seed",
      pto.oplib.seed_id = "seed_vec_bin_core",
      pto.oplib.seed_dtype = "f32",
      pto.oplib.seed.support_dtypes = ["f16", "f32"],
      pto.oplib.seed.support_ops = ["tadd", "tsub", "tmul", "tdiv", "tmax", "tmin"],
      pto.oplib.seed.core_slot = "binary_ewise_core",
      pto.oplib.match.rows = -1 : i64,
      pto.oplib.match.cols = -1 : i64,
      pto.oplib.match.blayout = "row_major",
      pto.oplib.match.slayout = "any",
      pto.oplib.match.fractal = -1 : i64,
      pto.oplib.cost = 10 : i64,
      pto.oplib.priority = 0 : i64,
      pto.simd.level = "binary_ewise_v1",
      pto.simd.lanes = 64 : i64
    } {
  %m0 = builtin.unrealized_conversion_cast %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>
  %m1 = builtin.unrealized_conversion_cast %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>
  %md = builtin.unrealized_conversion_cast %dst  : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>

  %flat0 = memref.reinterpret_cast %m0 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
  %flat1 = memref.reinterpret_cast %m1 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
  %flatd = memref.reinterpret_cast %md to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>

  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %c1024 step %c64 {
    %remain = arith.subi %c1024, %i : index
    %mask = pto.simd.predicate %remain : index -> vector<64xi1>

    %a = pto.simd.load %flat0, %i, %mask : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> vector<64xf32>
    %b = pto.simd.load %flat1, %i, %mask : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> vector<64xf32>

    // seed 的唯一核心槽位；实例化会把 addf 改写为目标 op。
    %c = arith.addf %a, %b {pto.simd.core_slot = "binary_ewise_core"} : vector<64xf32>

    pto.simd.store %c, %flatd, %i, %mask : vector<64xf32>, memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1>
  }
  return
}
```

## 5. 示例模板约束检查表（CR/自测直接打勾）

| # | 检查项 | 必须规则 | 常见失败码 |
| --- | --- | --- | --- |
| 1 | 外部 ABI | 输入 3 个 `!pto.tile_buf`，返回 `()` | `E_OPLIB_EMPTY_BODY_FOR_SIMD`（签名或模板结构不符合时常伴随） |
| 2 | SIMD 级别属性 | `pto.simd.level = "binary_ewise_v1"` | `E_OPLIB_SIMD_INVALID_CORE_SLOT`（level 不支持时） |
| 3 | lane 属性 | `pto.simd.lanes` 为正整数 | `E_OPLIB_SIMD_LANES_MISMATCH` |
| 4 | 模板体非空 | 函数非 external 且 body 非空 | `E_OPLIB_EMPTY_BODY_FOR_SIMD` |
| 5 | dtype 范围 | 仅 `f16/f32` | `E_OPLIB_SIMD_UNSUPPORTED_DTYPE` |
| 6 | layout 范围 | 仅 `row_major` | `E_OPLIB_SIMD_UNSUPPORTED_LAYOUT` |
| 7 | 核心槽位唯一性 | 恰好一个 `pto.simd.core_slot = "binary_ewise_core"` | `E_OPLIB_SIMD_INVALID_CORE_SLOT` |
| 8 | 核心 op 类型 | 必须是 `arith.addf/subf/mulf/divf/maximumf/minimumf` 之一 | `E_OPLIB_SIMD_INVALID_CORE_SLOT` |
| 9 | 顺序约束 | 满足 `load -> core -> store` | `E_OPLIB_SIMD_INVALID_CORE_SLOT` |
| 10 | lanes 一致性 | `predicate/load/store/core` 的向量 lane 与 `pto.simd.lanes` 一致 | `E_OPLIB_SIMD_LANES_MISMATCH` |
| 11 | `load_pu/store_pu` 步长 | `step > 0` | verifier 直接报错 |
| 12 | seed 改写约束 | seed 实例化时只能改写 core slot 对应算术 | `E_OPLIB_SIMD_INVALID_CORE_SLOT` |
| 13 | 实例函数必须有 body | 禁止 fake-body fallback | `E_OPLIB_INSTANCE_BODY_MISSING` |

## 6. 最小自检命令

1. 导入与选择阶段检查（看是否能产生实例）：

```bash
ptoas test/tile_fusion/softmax_chain.pto \
  --op-lib-dir=<your-oplib-dir> \
  --dump-ir-after-oplib-lowering -o -
```

2. 融合后检查（`pto.simd.*` 应已降到标准 dialect）：

```bash
ptoas test/tile_fusion/softmax_chain.pto \
  --op-lib-dir=<your-oplib-dir> \
  --dump-ir-after-op-fusion -o -
```

3. 快速定位常见失败：

```bash
ptoas <input.pto> --op-lib-dir=<your-oplib-dir> -o /tmp/out.cpp 2>&1 | \
  rg "E_OPLIB_|no matching OP-Lib entry|error:"
```

## 7. 高频踩坑

1. `pto.oplib.match.blayout` 写成 `row_major`，但模板匹配目标 tile 不是 row_major。
2. seed 里打了多个 `pto.simd.core_slot`，实例化改写会失败。
3. `pto.simd.lanes=64`，但写成 `vector<32xf32>` 或 `vector<128xf32>`。
4. 空模板体还期望走历史 fallback（V1 已明确禁止）。
5. 在模板体里忘记 `load -> core -> store` 主干顺序。

