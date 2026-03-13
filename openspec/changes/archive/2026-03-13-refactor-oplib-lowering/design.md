# Design: OpLib Lowering 重构

## 0. 审阅收敛（Phase 0）

### 0.1 `buildMatchRequest` 基线映射表（当前行为）
来源：`lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp` 中 `buildMatchRequest` 及其 helper（约 2299-2594 行）。

记号说明：
- `T` = Tile 操作数（`TemplateArgRole::Tile`）
- `S` = Scalar 操作数（`TemplateArgRole::Scalar`）

#### A. Float Binary / Partial Binary

| Op | kind | opName | operand 角色 | 特殊匹配字段 |
|---|---|---|---|---|
| `TMulOp` | `l3_float_binary_elementwise_template` | `tmul` | `src0:T, src1:T, dst:T` | 无 |
| `TDivOp` | `l3_float_binary_elementwise_template` | `tdiv` | `src0:T, src1:T, dst:T` | 无 |
| `TAddOp` | `l3_float_binary_elementwise_template` | `tadd` | `src0:T, src1:T, dst:T` | 无 |
| `TSubOp` | `l3_float_binary_elementwise_template` | `tsub` | `src0:T, src1:T, dst:T` | 无 |
| `TMaxOp` | `l3_float_binary_elementwise_template` | `tmax` | `src0:T, src1:T, dst:T` | 无 |
| `TMinOp` | `l3_float_binary_elementwise_template` | `tmin` | `src0:T, src1:T, dst:T` | 无 |
| `TPartAddOp` | `l3_float_partial_binary_template` | `tpartadd` | `src0:T, src1:T, dst:T` | 无 |
| `TPartMaxOp` | `l3_float_partial_binary_template` | `tpartmax` | `src0:T, src1:T, dst:T` | 无 |
| `TPartMinOp` | `l3_float_partial_binary_template` | `tpartmin` | `src0:T, src1:T, dst:T` | 无 |

#### B. Tile-Scalar / Ternary

| Op | kind | opName | operand 角色 | 特殊匹配字段 |
|---|---|---|---|---|
| `TAddSOp` | `l3_float_tile_scalar_template` | `tadds` | `src:T, scalar:S, dst:T` | `scalarPos=1` |
| `TSubSOp` | `l3_float_tile_scalar_template` | `tsubs` | `src:T, scalar:S, dst:T` | `scalarPos=1` |
| `TMulSOp` | `l3_float_tile_scalar_template` | `tmuls` | `src0:T, scalar:S, dst:T` | `scalarPos=1` |
| `TMaxSOp` | `l3_float_tile_scalar_template` | `tmaxs` | `src:T, scalar:S, dst:T` | `scalarPos=1` |
| `TMinSOp` | `l3_float_tile_scalar_template` | `tmins` | `src:T, scalar:S, dst:T` | `scalarPos=1` |
| `TDivSOp` | `l3_float_tile_scalar_template` | `tdivs` | `src:T, scalar:S, dst:T` | `scalarPos=1`，`requiredVariantId=pto.tdivs.order`（`tile_scalar`/`scalar_tile`） |
| `TAddCOp` | `l3_float_ternary_tile_template` | `taddc` | `src0:T, src1:T, src2:T, dst:T` | 无 |
| `TSubCOp` | `l3_float_ternary_tile_template` | `tsubc` | `src0:T, src1:T, src2:T, dst:T` | 无 |
| `TAddSCOp` | `l3_float_ternary_tile_scalar_template` | `taddsc` | `src0:T, scalar:S, src1:T, dst:T` | `scalarPos=1` |
| `TSubSCOp` | `l3_float_ternary_tile_scalar_template` | `tsubsc` | `src0:T, scalar:S, src1:T, dst:T` | `scalarPos=1` |
| `TLReluOp` | `l3_float_tile_scalar_template` | `tlrelu` | `src:T, slope:S, dst:T` | `scalarPos=1` |
| `TPReluOp` | `l3_float_ternary_tile_template` | `tprelu` | `src0:T, src1:T, tmp:T, dst:T` | `tmp` 必选；`tmp` 元素类型必须 `i8` |

#### C. Unary

| Op | kind | opName | operand 角色 | 特殊匹配字段 |
|---|---|---|---|---|
| `TAbsOp` | `l3_float_unary_template` | `tabs` | `src:T, dst:T` | 无 |
| `TNegOp` | `l3_float_unary_template` | `tneg` | `src:T, dst:T` | 无 |
| `TRecipOp` | `l3_float_unary_template` | `trecip` | `src:T, dst:T` | 无 |
| `TReluOp` | `l3_float_unary_template` | `trelu` | `src:T, dst:T` | 无 |
| `TExpOp` | `l3_float_unary_math_template` | `texp` | `src:T, dst:T` | 无 |
| `TLogOp` | `l3_float_unary_math_template` | `tlog` | `src:T, dst:T` | 无 |
| `TSqrtOp` | `l3_float_unary_math_template` | `tsqrt` | `src:T, dst:T` | 无 |
| `TRsqrtOp` | `l3_float_unary_math_template` | `trsqrt` | `src:T, dst:T` | 无 |
| `TNotOp` | `l3_int_unary_template` | `tnot` | `src:T, dst:T` | 无 |

#### D. Reduction / Broadcast / Expand

| Op | kind | opName | operand 角色 | 特殊匹配字段 |
|---|---|---|---|---|
| `TRowSumOp` | `l3_reduce_row_template` | `trowsum` | `src:T, tmp:T, dst:T` | `tmp` 必选 |
| `TRowMaxOp` | `l3_reduce_row_template` | `trowmax` | `src:T, tmp:T, dst:T` | `tmp` 必选 |
| `TRowMinOp` | `l3_reduce_row_template` | `trowmin` | `src:T, tmp:T, dst:T` | `tmp` 必选 |
| `TColMaxOp` | `l3_reduce_col_template` | `tcolmax` | `src:T, dst:T` | 无 |
| `TColMinOp` | `l3_reduce_col_template` | `tcolmin` | `src:T, dst:T` | 无 |
| `TColSumOp` | `l3_reduce_colsum_template` | `tcolsum` | `src:T, tmp:T, dst:T` | `tmp` 必选；`isBinary=colsum.getIsBinary()` |
| `TRowExpandOp` | `l3_broadcast_row_template` | `trowexpand` | `src:T, dst:T` | 无 |
| `TColExpandOp` | `l3_broadcast_col_template` | `tcolexpand` | `src:T, dst:T` | 无 |
| `TRowExpandMulOp` | `l3_broadcast_row_binary_template` | `trowexpandmul` | `src0:T, src1:T, dst:T` | 无 |
| `TRowExpandDivOp` | `l3_broadcast_row_binary_template` | `trowexpanddiv` | `src0:T, src1:T, dst:T` | 无 |
| `TRowExpandSubOp` | `l3_broadcast_row_binary_template` | `trowexpandsub` | `src0:T, src1:T, dst:T` | 无 |
| `TExpandsOp` | `l3_scalar_expand_template` | `texpands` | `scalar:S, dst:T` | `scalarPos=0` |

#### E. Compare / Select / Bitwise

| Op | kind | opName | operand 角色 | 特殊匹配字段 |
|---|---|---|---|---|
| `TCmpOp` | `l3_cmp_tile_tile_template` | `tcmp` | `src0:T, src1:T, dst:T` | `cmpMode`（缺省 `EQ`） |
| `TCmpSOp` | `l3_cmp_tile_scalar_template` | `tcmps` | `src:T, scalar:S, dst:T` | `scalarPos=1`；`cmpMode`（缺省 `EQ`） |
| `TSelOp` | `l3_select_mask_template` | `tsel` | `mask:T, src0:T, src1:T, dst:T` | `mask` 元素类型要求 `i32` |
| `TSelSOp` | `l3_select_scalar_template` | `tsels` | `src0:T, src1:T, selectMode:S, dst:T` | `scalarPos=2` |
| `TAndOp` | `l3_int_binary_elementwise_template` | `tand` | `src0:T, src1:T, dst:T` | 无 |
| `TOrOp` | `l3_int_binary_elementwise_template` | `tor` | `src0:T, src1:T, dst:T` | 无 |
| `TXorOp` | `l3_int_binary_elementwise_template` | `txor` | `src0:T, src1:T, dst:T` | 无 |
| `TShlOp` | `l3_int_binary_elementwise_template` | `tshl` | `src0:T, src1:T, dst:T` | 无 |
| `TShrOp` | `l3_int_binary_elementwise_template` | `tshr` | `src0:T, src1:T, dst:T` | 无 |
| `TAndSOp` | `l3_int_tile_scalar_elementwise_template` | `tands` | `src:T, scalar:S, dst:T` | `scalarPos=1` |
| `TOrSOp` | `l3_int_tile_scalar_elementwise_template` | `tors` | `src:T, scalar:S, dst:T` | `scalarPos=1` |
| `TXorSOp` | `l3_int_tile_scalar_elementwise_template` | `txors` | `src:T, scalar:S, dst:T` | `scalarPos=1` |
| `TShlSOp` | `l3_int_tile_scalar_elementwise_template` | `tshls` | `src:T, scalar:S, dst:T` | `scalarPos=1` |
| `TShrSOp` | `l3_int_tile_scalar_elementwise_template` | `tshrs` | `src:T, scalar:S, dst:T` | `scalarPos=1` |

#### F. Remainder

| Op | kind | opName | operand 角色 | 特殊匹配字段 |
|---|---|---|---|---|
| `TRemOp` | `l3_float_binary_elementwise_template` | `trem` | `src0:T, src1:T, dst:T` | 仅支持 float dtype（否则 `failure()`） |
| `TRemSOp` | `l3_float_tile_scalar_template` | `trems` | `src:T, scalar:S, dst:T` | `scalarPos=1`；仅支持 float dtype（否则 `failure()`） |

基线统计：`buildMatchRequest` 当前覆盖 58 个 PTO Op 分支。

### 0.2 接口契约必须覆盖字段（Phase 0 要求）
接口化后，`OpLibOpInterface` 提供的信息必须能无损构造 `MatchRequest`，最小字段如下：

- `kind`：模板类别（如 `l3_float_binary_elementwise_template`）。
- `opName`：模板内部 op 键（如 `tadd`、`trowsum`）。
- `operand roles`：每个参与匹配的 operand 顺序与角色（`T/S`），含可选 `tmp`。
- `scalarPos`：用于 tile-scalar / scalar-expand / select-scalar 等模板匹配。
- `cmpMode`：比较类模板匹配键（默认值策略也需保留）。
- `isBinary`：`tcolsum` 等分支的匹配键。
- `requiredVariantId`：`tdivs` 的 `tile_scalar/scalar_tile` 选择约束。

### 0.3 本次重构边界（Phase 0 要求）

In Scope：
- 将 `buildMatchRequest` 的算子分发入口改为接口驱动；
- 保持现有匹配语义（字段含义、默认值、失败条件）不变；
- 允许迁移期保留 legacy fallback 以降低回归风险。

Out of Scope：
- 不改 `TemplateRegistry::selectVariantFor` / `getOrCreateInstance` 的选择语义；
- 不改 `pto.oplib.*` 与 `pto.oplib.instance.*` attribute schema；
- 不改变已有 OpLib 模板文件内容和 symbol 命名策略。

### 0.4 完成标准（Phase 0 要求）

满足以下条件，视为 Phase 0 完成：
- 已形成并落盘本基线映射表（`op -> kind/opName/operand 角色/特殊匹配字段`）；
- 接口契约字段集合明确包含：`scalarPos/cmpMode/isBinary/requiredVariantId/tmp`；
- 文档明确了 In Scope / Out of Scope；
- 明确验收口径：所有当前可 lowering 的 58 个 Op，均可经接口路径组装等价 `MatchRequest`。

## 1. 详细设计 (Detailed Design)

### 1.1 接口定义 (`OpLibOpInterface`)
在 `PTOInterfaces.td` 中新增 `OpLibOpInterface`，采用单方法返回描述结构，避免 Pass 侧拼装时需要跨多个 getter 组装状态。

#### 1.1.1 ABI 选择
- 新增结构体：`mlir::pto::OpLibMatchDescriptor`（定义于 `PTO.h`）。
- 接口方法：
  - `FailureOr<OpLibMatchDescriptor> getOpLibMatchDescriptor()`
- `OpLibMatchDescriptor` 字段：
  - `kind/opName`
  - `operands/operandRoles`
  - `scalarPos/cmpMode/isBinary/requiredVariantId`

该 ABI 可以一一覆盖 Phase 0 中的 `MatchRequest` 关键字段，并支持后续在 op 侧做细粒度失败诊断（返回 `failure()`）。

#### 1.1.2 兼容策略
- 迁移初期保留 `buildMatchRequest` 旧路径；
- 新增 `buildMatchRequestFromInterface`，优先走 `OpLibOpInterface`；
- 对未实现接口的 op 回退到 legacy 分支；
- 当 A-D 批次算子全部完成后删除 legacy 分支。

### 1.2 算子适配
按任务分批为 Op 挂载接口实现，迁移过程中保证与 0.1 基线映射一致。

### 1.3 Pass 逻辑重构
- 遍历函数中的所有 Op；
- 检查是否实现了 `OpLibOpInterface`；
- 从接口信息构建 `MatchRequest` 并走现有 `TemplateRegistry` 路径；
- 迁移后再删除 legacy `dyn_cast` 大分支。

### 1.4 迁移顺序与收敛条件（实施状态）
- 迁移顺序：先完成算子侧 A-D 批次接口实现，再切换 Pass 入口到 `buildMatchRequestFromInterface`。
- legacy 移除条件：当 A-D 批次覆盖完成且回归用例通过后，删除 `buildMatchRequest` 的 `dyn_cast` 大分支。
- 当前状态（2026-03-13）：已满足移除条件，`PTOLowerToOpLibCalls.cpp` 仅保留接口驱动入口；fusion group 与 single-op 路径统一复用同一 `planOneOpLowering -> buildMatchRequestFromInterface` 入口。

### 1.5 严格门禁（4.5~4.9 必须走 OP-Lib）

新增约束（面向 `docs/PTO_IR_manual.md` 第 4.5~4.9 节）：

- 所有属于 4.5~4.9 family 的 PTO op（当前以 `isSupportedFusionOp` 集合为准）必须被 `PTOInstantiateAndLowerToLibCall` 改写为 OP-Lib call。
- 不再允许这批 op 走“single-op fallback / fusion-group fallback”保留原 op。
- 任一 op 在匹配、选择、实例化、重写任一阶段失败，pass 必须 `emitError` 并失败退出。
- pass 末尾执行一致性检查：若函数中仍残留 4.5~4.9 family 的 PTO op，直接报错。

实现要点：

- `planOneOpLowering` 的失败诊断从 warning 升级为 error。
- `lowerOneGroup/lowerSingleOps` 在 plan/rewrite 失败时立即返回 `failure()`，不再 `continue`。
- 新增 `verifyNoRemainingTargetOps` 收口检查，避免“带 group attr 但未进入 rewrite”的漏网情况。

### 1.6 修复方案（面向当前 build/output 暴露问题）

基于 `2026-03-13` 对 `build/output_log` 的审计（命令：`rg -uu "error: no matching OP-Lib entry"`），当前 4.5~4.9 相关缺口如下：

| 缺口簇 | 受影响 op | 典型日志 | 当前根因 |
|---|---|---|---|
| Int bitwise dtype 缺口 | `tand/tor/txor/tnot/tands/tors/txors` | `build/output_log/And/and-ptoas.log`、`.../Or/or-ptoas.log`、`.../Xor/xor-ptoas.log` | `oplib/level3/int_*` 模板仅提供 `i32`，未覆盖 `i16` |
| Float binary dtype 缺口 | `tadd/tsub` | `build/output_log/Sync/test_inject_sync_if-ptoas.log` 等 | `l3_float_binary_elementwise_template` seed 仅 `f32` |
| Reduce/Broadcast row 布局缺口 | `trowmax/trowsum/trowexpandmul/trowexpanddiv/trowexpandsub` | `build/output_log/Complex/mix_kernel-ptoas.log`、`.../PyPTOIRParser/*softmax*.log` | 模板匹配要求 row-major 行向量输入/输出，样例为 `16x1 col_major` |

补充观察：

- strict gate 在当前二进制中已生效：`build/tools/ptoas/ptoas build/output/And/and-pto-ir.pto ...` 返回 `exit=1` 且不生成输出 `.cpp`。
- 现有 `build/output/*.cpp` 中的直连 `T*` 不能单独作为判据（可能是历史生成物）；以 `no matching OP-Lib entry` 诊断为主。

修复顺序（按收益/风险）：

1. `dtype` 扩展第一批（高优先）
- 补齐 `int_binary/int_unary/int_tile_scalar_elementwise` 的 `i16` 变体（覆盖 `tand/tor/txor/tnot/tands/tors/txors`）。
- 补齐 `float_binary_elementwise` 的 `f16` 变体（先覆盖 `tadd/tsub`，再评估是否一次补齐 `tmul/tdiv/tmax/tmin`）。
- 每批补齐后回归 `test/oplib` + 对应 `build/output` 样例，确认由“报错”转为“命中 OP-Lib lowering”。

2. Row reduce/broadcast 布局变体（高优先）
- 为 `l3_reduce_row_template` 与 `l3_broadcast_row_binary_template` 增加 `arg1/arg2` 支持 `col_major 16x1` 的匹配变体，或在 pass 前增加等价 reshape/bridge。
- 保持 `TemplateRegistry` 选择语义不变，仅增加可匹配 entry。

3. Compare/select 非同构 operand 风险收敛（中优先）
- 统一 `tcmp/tcmps/tsel` mask 类型契约（`i32` 与 `i8/ui8` 策略二选一并固化）。
- 同步 verifier + OpLib 模板 + emitc/lit 断言，避免接口路径与模板约束分叉。

4. 收口验收（必须）
- `build/output_log` 不再出现 4.5~4.9 相关 `no matching OP-Lib entry`。
- strict gate 保持开启；任一新引入缺口继续 fail-fast。

## 2. 交付物 (Deliverables)
- 接口定义文件（`PTOInterfaces.td`）；
- 重构后的算子接口实现（`PTOOps.td` / `PTO.cpp`）；
- 接口驱动的 `PTOLowerToOpLibCalls.cpp`。
