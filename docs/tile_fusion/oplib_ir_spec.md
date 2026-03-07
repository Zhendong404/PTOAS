# Level-3 OP-Lib IR 接口规范（V1.1：Mixed Body IR）

- 状态：Draft v1.1
- 生效范围：PTOAS OP-Lib Binary Element-Wise 主链路
- 目标读者：PTOAS OP Fusion 维护者、OP-Lib 开发者

## 1. 设计目标与范围

### 1.1 目标

本文定义 Level-3 OP-Lib 在 Binary Element-Wise 场景的模板体 IR 规范，目标如下：

1. 保持 OP-Lib 对外 ABI 不变：模板/实例/调用点均为 `!pto.tile_buf`。
2. 将 `pto.simd`、`vector`、`arith` 统一纳入 OP-Lib 开发者可写 IR 集合。
3. 不再要求主链路必须执行 `pto.simd -> vector` 专用 lowering。
4. 保持 `variant/seed` 选择机制与 `pto.oplib.*` 元数据兼容。
5. 为后续 EmitC/CCEC 路径保留更直接的代码生成空间。

### 1.2 V1.1 范围

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

### 2.2 扩展（函数体语义层）

模板函数体采用 Mixed Body IR：

1. 可直接使用 `vector/arith/scf/memref` 表达核心计算与循环。
2. 可选使用 `pto.simd.*` 表达显式 lane/mask 语义。
3. `pto.simd.*` 出现时，必须提供对应 `pto.simd` 属性。

## 3. 模板体可用 IR 集合

### 3.1 必选约束

1. 入口签名固定：`(!pto.tile_buf, !pto.tile_buf, !pto.tile_buf) -> ()`。
2. 模板体不能为空。
3. 不允许落到集合外 dialect/op（避免不可控语义漂移）。

### 3.2 允许的 IR（V1.1）

1. `arith.*`
2. `vector.*`
3. `memref.*`
4. `scf.*`
5. `builtin.unrealized_conversion_cast`（仅用于 `tile_buf <-> memref` 桥接）
6. `pto.simd.*`（可选）

说明：`pto` 其他高层 tile 计算 op（如 `pto.tadd/pto.tmul`）不应再次出现在 OP-Lib 模板体内。

## 4. `pto.simd.*` 语义（可选层）

当模板需要显式 mask/predicate/post-update 语义时，可使用 `pto.simd.*`：

1. `pto.simd.predicate`
2. `pto.simd.load`
3. `pto.simd.store`
4. `pto.simd.load_pu`
5. `pto.simd.store_pu`

### 4.1 `pto.simd` 属性规则（仅在使用 `pto.simd.*` 时强制）

1. `pto.simd.level = "binary_ewise_v1"`
2. `pto.simd.lanes = <i64>`
3. `pto.simd.core_slot = "binary_ewise_core"`（标在核心 `arith.*` op 上）

### 4.2 纯 vector 模板

纯 `vector/arith` 模板允许不写 `pto.simd.level/lanes`，但仍必须满足 V1.1 的 dtype/layout/seed 规则。

## 5. Seed 与 Core Slot 规则

1. `seed` 函数体必须且仅有一个核心 slot op。
2. 核心 slot 默认标记为 `pto.simd.core_slot = "binary_ewise_core"`。
3. 核心 slot op 必须是以下之一：
   1. `arith.addf`
   2. `arith.subf`
   3. `arith.mulf`
   4. `arith.divf`
   5. `arith.maximumf`
   6. `arith.minimumf`
4. 实例化时仅改写核心 slot op，访存/循环骨架保持不变。
5. 映射关系：
   1. `tadd -> arith.addf`
   2. `tsub -> arith.subf`
   3. `tmul -> arith.mulf`
   4. `tdiv -> arith.divf`
   5. `tmax -> arith.maximumf`
   6. `tmin -> arith.minimumf`

## 6. 校验规则（硬失败）

### 6.1 模板导入校验

1. 签名必须符合 OP-Lib ABI。
2. 函数体必须非空（不允许空模板体 fallback）。
3. dtype 仅允许 `f16/f32`。
4. layout 仅允许 `row_major`。
5. `seed` 必须满足核心 slot 唯一性与类型合法性。
6. 若函数体包含 `pto.simd.*`，必须满足：
   1. `pto.simd.level/lanes` 存在且合法。
   2. 所有相关 vector lane 与 `pto.simd.lanes` 一致。

### 6.2 错误码

1. `E_OPLIB_EMPTY_BODY_FOR_SIMD`
2. `E_OPLIB_SIMD_LANES_MISMATCH`
3. `E_OPLIB_SIMD_INVALID_CORE_SLOT`
4. `E_OPLIB_SIMD_UNSUPPORTED_DTYPE`
5. `E_OPLIB_SIMD_UNSUPPORTED_LAYOUT`
6. `E_OPLIB_INSTANCE_BODY_MISSING`
7. `E_OPLIB_BODY_DISALLOWED_IR`
8. `E_OPLIB_SIMD_ATTR_REQUIRED`

## 7. Pass 行为约束（V1.1）

1. `PTOInstantiateAndLowerToLibCallPass`：
   1. 导入模板时执行 Mixed Body IR 校验。
   2. 创建实例时克隆模板体（不再创建空 body 再回填）。
   3. Seed 实例执行 core slot 算术改写。
2. `PTOInlineLibCallPass`：
   1. 不再允许 fake body fallback。
   2. 实例函数若无函数体，直接 `E_OPLIB_INSTANCE_BODY_MISSING`。
3. `PTOValidateSimdIRPass`：
   1. 仅对包含 `pto.simd.*` 的函数执行结构合法性校验。
4. 主链路不再依赖 `PTOLowerSimdToVectorPass` 作为必经步骤。

## 8. 最小示例（两种写法）

### 8.1 纯 vector 写法（无 `pto.simd.*`）

```mlir
// scf.for + vector.load/store + arith.addf
// 适合直接走向 EmitC/CCEC 可控映射路径
```

### 8.2 `pto.simd` 写法（显式 mask 语义）

```mlir
// pto.simd.predicate/load/store + arith.addf(core_slot)
// 适合需要表达 lane mask / 尾块语义的模板
```

## 9. 开发者指南

更多“最小可用模板写法 + 约束检查表”见：

- `docs/tile_fusion/oplib_simd_template_min_guide.md`
