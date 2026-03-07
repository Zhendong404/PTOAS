# Level-3 OP-Lib IR 接口规范（V1：`pto.simd.*`）

- 状态：Draft v1.0
- 生效范围：PTOAS OP-Lib Binary Element-Wise 主链路
- 目标读者：PTOAS OP Fusion 维护者、OP-Lib 开发者

## 1. 设计目标与范围

### 1.1 目标

本文定义 Level-3 OP-Lib 在 Binary Element-Wise 场景的细粒度 IR 规范，目标如下：

1. 保持 OP-Lib 对外 ABI 不变：模板/实例/调用点均为 `!pto.tile_buf`。
2. 在模板函数体内引入 `pto.simd.*` 语义，显式表达 tile 到 SIMD/lane 级桥接。
3. 复用 `vector/arith/scf/memref`，避免重复定义计算语义。
4. 保持 `variant/seed` 选择机制与 `pto.oplib.*` 元数据兼容。

### 1.2 V1 范围

1. OP：`tadd/tsub/tmul/tdiv/tmax/tmin`
2. dtype：`f16/f32`
3. layout：`row_major`
4. 只覆盖 Binary Element-Wise，不覆盖 reduce/expand/scalar 变体。

## 2. 双层契约

### 2.1 保持不变（匹配/选择层）

以下元数据前缀继续使用 `pto.oplib.*`：

1. `pto.oplib.kind`
2. `pto.oplib.entry_role`
3. `pto.oplib.op`
4. `pto.oplib.variant_id`
5. `pto.oplib.match.*`
6. `pto.oplib.cost` / `pto.oplib.priority`
7. `pto.oplib.seed.*`

### 2.2 新增（函数体语义层）

模板函数体必须使用 `pto.simd.*` 与标准 `arith/vector/scf` 组合表达核心逻辑。

新增函数属性：

1. `pto.simd.level = "binary_ewise_v1"`
2. `pto.simd.lanes = <i64>`

新增核心槽位属性（标在核心算术 op 上）：

1. `pto.simd.core_slot = "binary_ewise_core"`

## 3. `pto.simd.*` 原语（V1）

### 3.1 SIMD 谓词

```mlir
%mask = pto.simd.predicate %active : index -> vector<64xi1>
```

语义：生成 lane 掩码，`lane < active` 为 true。

### 3.2 SIMD 加载/存储

```mlir
%v = pto.simd.load %src, %off, %mask
     : memref<1024xf32, ...>, index, vector<64xi1> -> vector<64xf32>

pto.simd.store %v, %dst, %off, %mask
  : vector<64xf32>, memref<1024xf32, ...>, index, vector<64xi1>
```

语义：按 mask 执行 lane 级读写。

### 3.3 带 post-update 的加载/存储

```mlir
%v, %next = pto.simd.load_pu %src, %off, %mask {step = 64 : i64}
            : memref<1024xf32, ...>, index, vector<64xi1>
              -> vector<64xf32>, index

%next2 = pto.simd.store_pu %v, %dst, %off, %mask {step = 64 : i64}
         : vector<64xf32>, memref<1024xf32, ...>, index, vector<64xi1>
           -> index
```

语义：读/写后返回更新后的 offset。

## 4. Seed 与 Core Slot 规则

1. `seed` 函数体必须且仅有一个 `pto.simd.core_slot = "binary_ewise_core"` 的核心算术 op。
2. 实例化时仅改写该 core op，其他访存/循环骨架保持不变。
3. 映射关系：
   1. `tadd -> arith.addf`
   2. `tsub -> arith.subf`
   3. `tmul -> arith.mulf`
   4. `tdiv -> arith.divf`
   5. `tmax -> arith.maximumf`
   6. `tmin -> arith.minimumf`

## 5. 校验规则（硬失败）

### 5.1 模板导入校验

1. 必须提供 `pto.simd.level` 与 `pto.simd.lanes`。
2. 函数体不能为空（不允许空模板体）。
3. 必须包含且仅包含一个核心 slot op。
4. 必须满足 `load -> core -> store` 顺序。

### 5.2 错误码

1. `E_OPLIB_EMPTY_BODY_FOR_SIMD`
2. `E_OPLIB_SIMD_LANES_MISMATCH`
3. `E_OPLIB_SIMD_INVALID_CORE_SLOT`
4. `E_OPLIB_SIMD_UNSUPPORTED_DTYPE`
5. `E_OPLIB_SIMD_UNSUPPORTED_LAYOUT`
6. `E_OPLIB_INSTANCE_BODY_MISSING`

## 6. Pass 行为约束（V1）

1. `PTOInstantiateAndLowerToLibCallPass`：
   1. 导入模板时执行 SIMD 属性与函数体校验。
   2. 创建实例时克隆模板体（不再创建空 body 再回填）。
   3. Seed 实例执行 core slot 算术改写。
2. `PTOInlineLibCallPass`：
   1. 不再允许 fake body fallback。
   2. 实例函数若无函数体，直接 `E_OPLIB_INSTANCE_BODY_MISSING`。
3. `PTOValidateSimdIRPass`：
   1. 统一校验 `pto.simd.*` 结构合法性。
4. `PTOLowerSimdToVectorPass`：
   1. 将 `pto.simd.*` 降到 `vector/memref/arith`。

## 7. 模板编写建议

1. 统一以 `tile_buf` 入参，在函数体内部桥接到 `memref`。
2. SIMD 处理建议使用线性 offset（例如把 2D memref reinterpret 为 1D）。
3. 核心算术统一使用 `arith.*` 的 vector 形式，避免自定义计算 op。
4. `variant` 和 `seed` 都建议显式写出 `pto.simd.level` / `pto.simd.lanes`。

## 8. 最小示例（seed 骨架）

```mlir
func.func private @__pto_oplib_seed_vec_bin_core(
  %src0: !pto.tile_buf<...>,
  %src1: !pto.tile_buf<...>,
  %dst:  !pto.tile_buf<...>
) attributes {
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
  // load -> core(slot) -> store
  // core op should carry: { pto.simd.core_slot = "binary_ewise_core" }
  return
}
```

## 9. 开发者指南

更多“最小可用模板写法 + 约束检查表”见：

- `docs/tile_fusion/oplib_simd_template_min_guide.md`
