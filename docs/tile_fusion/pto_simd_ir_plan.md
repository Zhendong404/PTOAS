# OP-Lib 模板体 IR V1.1（Mixed IR Authoring）落地方案

## Summary
1. 保持现有 OP-Lib 外部 ABI 不变：模板/实例/调用点继续使用 `(!pto.tile_buf, !pto.tile_buf, !pto.tile_buf) -> ()`。
2. 把 `pto.simd`、`vector`、`arith` 统一定义为 OP-Lib 开发者可用 IR。
3. `variant/seed` 匹配与选择元数据继续保留 `pto.oplib.*`。
4. V1.1 范围锁定 Binary Element-Wise：`tadd/tsub/tmul/tdiv/tmax/tmin`，数据类型 `f16/f32`，布局 `row_major`。
5. 取消“必须 `PTOLowerSimdToVector`”要求，改为“可选 `pto.simd` + 直接 `vector` 写法并存”。

## Interface Changes
1. 模板体可用 IR 集合（allowlist）：
   1. `arith.*`
   2. `vector.*`
   3. `memref.*`
   4. `scf.*`
   5. `builtin.unrealized_conversion_cast`（仅桥接 `tile_buf <-> memref`）
   6. `pto.simd.*`（可选）
2. `pto.simd.*` 属性规则改为“按需强制”：
   1. 当且仅当模板体使用 `pto.simd.*` 时，必须提供：
      1. `pto.simd.level = "binary_ewise_v1"`
      2. `pto.simd.lanes = <i64>`
      3. `pto.simd.core_slot = "binary_ewise_core"`（标在核心 `arith.*` op 上）
3. 保持不变的匹配元数据前缀：
   1. `pto.oplib.kind`
   2. `pto.oplib.entry_role`
   3. `pto.oplib.op`
   4. `pto.oplib.variant_id`
   5. `pto.oplib.match.*`
   6. `pto.oplib.cost` / `pto.oplib.priority`
   7. `pto.oplib.seed.*`
4. Seed 规则保持单 core slot：
   1. 每个 seed 体必须且仅有一个核心算术 op 槽位。
   2. 实例化仅替换该核心算术语义：`add/sub/mul/div/maximum/minimum`。

## Implementation Changes
1. `docs/tile_fusion/oplib_ir_spec.md` 改为 Mixed Body IR 规范，明确“`vector/arith/pto.simd` 均可直接编写”。
2. `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp` 扩展模板体验证逻辑：
   1. 校验 body 非空、ABI 合法、V1.1 范围合法（f16/f32、row_major）。
   2. 校验 body 仅使用 allowlist IR。
   3. 若 body 使用 `pto.simd.*`，额外校验 `pto.simd.level/lanes/core_slot`。
3. `lib/PTO/Transforms/PTOInstantiateAndInlineOpLib.cpp` 继续禁用 fake body fallback：
   1. 实例函数必须来自模板体克隆与 seed 改写。
   2. 无 body 直接失败，诊断包含 `variant_id/op/dtype`。
4. Pass 调整：
   1. `PTOValidateSimdIR` 仅负责“含 `pto.simd.*` 模板”的结构一致性检查。
   2. 移除主链路对 `PTOLowerSimdToVector` 的必经依赖。
   3. 如需研究/调试，可保留可选开关触发 `simd->vector` 降级，不影响默认代码生成路径。
5. 编译期错误码新增并统一文案：
   1. `E_OPLIB_EMPTY_BODY_FOR_SIMD`
   2. `E_OPLIB_SIMD_LANES_MISMATCH`
   3. `E_OPLIB_SIMD_INVALID_CORE_SLOT`
   4. `E_OPLIB_SIMD_UNSUPPORTED_DTYPE`
   5. `E_OPLIB_SIMD_UNSUPPORTED_LAYOUT`
   6. `E_OPLIB_INSTANCE_BODY_MISSING`
   7. `E_OPLIB_BODY_DISALLOWED_IR`
   8. `E_OPLIB_SIMD_ATTR_REQUIRED`

## Test Plan
1. 正例：
   1. `variant` 使用纯 `vector/arith` 模板体，成功选择、实例化、inline、后续融合。
   2. `variant` 使用 `pto.simd.*` 模板体，成功选择、实例化、inline。
   3. `seed` 单 core slot 覆盖六个 Binary op，实例化后核心算术替换正确。
2. 负例：
   1. 空模板体报 `E_OPLIB_EMPTY_BODY_FOR_SIMD`。
   2. 使用 allowlist 外 op/dialect 报 `E_OPLIB_BODY_DISALLOWED_IR`。
   3. 多 `core_slot` 或缺失 `core_slot` 报 `E_OPLIB_SIMD_INVALID_CORE_SLOT`。
   4. 使用 `pto.simd.*` 但缺少 `pto.simd.level/lanes` 报 `E_OPLIB_SIMD_ATTR_REQUIRED`。
   5. lanes 与向量宽度不一致报 `E_OPLIB_SIMD_LANES_MISMATCH`。
   6. 非 `f16/f32` 或非 `row_major` 报错。
3. 回归：
   1. `cost/priority/variant_id` 选择稳定不变。
   2. `pto.oplib.*` 元数据导入与 seed/variant 机制行为不变。
   3. `--dump-ir-after-oplib-lowering` 与 `--dump-ir-after-op-fusion` 可同时观测纯 vector 模板与 `pto.simd` 模板路径。

## Assumptions And Defaults
1. V1.1 只覆盖 Binary Element-Wise，不覆盖 reduce/expand/scalar 变体。
2. 语义规范不绑定具体 `pto-isa` 指令名，仅保证可等价映射。
3. 对外 ABI、融合组策略、pass 大体顺序保持现状，变更集中在模板体验证和模板体可用 IR 集。
4. 命名保持 `pto.simd.*`，但不再把其定位为唯一细粒度表达方式。
