## Stage-1：在 PTOAS 实现 `PTO OP -> OP-LIB call`（含 seed 自动实例化）

### Summary
- 目标：按 `oplib_ir_spec` 的接口落地 OP-LIB 调用转换，首批支持 `tadd/tsub/tmul/tdiv/tmax/tmin`。
- 范围：先在 **post-view（`PTOViewToMemref` 之后）** 实现，保证接口一致；OP-LIB 体可用 fake 实现。
- 关键策略（已锁定）：
  - `oplib-lowering` 默认开启；
  - `op-fusion` 依赖 `oplib-lowering`；
  - 单 OP 无匹配：回退原 OP + warning；
  - fusion 组内失配：整组回退 + warning；
  - dtype 先支持 `f16/f32`；
  - 未提供 `--op-lib-dir` 且启用 oplib-lowering 时直接报错。

### Key Implementation Changes
- **CLI 与 Pipeline 编排**
  - 新增开关：`--disable-oplib-lowering`（默认 false，即默认开启 lowering）。
  - 保留 `--enable-op-fusion`，并强制依赖 `oplib-lowering`（若显式关闭 lowering 但开启 fusion，直接报错）。
  - 运行顺序调整为：
    1. `LoweringSyncToPipe -> PTOViewToMemref`
    2. `InferPTOLayout/PlanMemory/InsertSync`（按现有条件）
    3. 若 `op-fusion` 开启：先做 fusion group 创建
    4. 统一执行 OP-LIB materialization（既处理 fusion 组，也处理非组单 OP）
    5. 执行 instantiate/inlining（保证实例函数可落地）
    6. 若 `op-fusion` 开启：保留现有 CSE/LoopFusion
  - 增加一个 IR 观测点（建议）：`--dump-ir-after-oplib-lowering`，用于新路径测试与回归。

- **OP-LIB 注册/匹配/实例化框架升级**
  - 将当前 OP-LIB 解析从 V1 旧属性升级为新规范：
    - 支持 `entry_role=variant|seed`
    - 支持 `seed_id/seed_dtype/support_dtypes/support_ops/core_slot`
    - 仍使用 `cost/priority` 做确定性选择
  - 匹配主键按 `op + dtype + shape + layout`：
    - `variant` 直接匹配
    - `seed` 先校验 `support_ops/support_dtypes`，再实例化候选
  - seed 自动实例化实现（关键）：
    - 从 seed 克隆实例函数
    - `dtype` 改写（`seed_dtype -> target dtype`）
    - 按 `target op` 替换 core slot 指令
      - `tadd/addf`、`tsub/subf`、`tmul/mulf`、`tdiv/divf`、`tmax/maximumf`、`tmin/minimumf`
    - 生成稳定实例标识（如 `__seed__<seed_id>__<op>__<dtype>`）并缓存复用
  - 候选统一排序：`cost ASC -> priority DESC -> variant_id LEX ASC`。

- **PTO OP 到 OP-LIB 调用改写**
  - 扩展支持 op 集：`tadd/tsub/tmul/tdiv/tmax/tmin`（包括分组逻辑与物化逻辑）。
  - 改写覆盖两类场景：
    1. **fusion 组内**：按组物化；若组内任一 OP 无匹配/实例化失败，整组回退且 warning。
    2. **非组单 OP**：逐个尝试改写为 OP-LIB call；无匹配则保留原 OP 且 warning。
  - 调用签名保持与当前 memref-level IR 一致（阶段性实现），同时保证 OP-LIB 属性与接口规范一致。

- **文档与规范同步**
  - 同步更新 `oplib_ir_spec` 的首批支持范围为 6 个 OP（加入 `tmax/tmin`）。
  - 明确阶段性实现说明：当前为 post-view 落地，后续再演进 pre-view。

### Test Plan
- **基础路径**
  1. `--disable-oplib-lowering` 未开启、且提供 `--op-lib-dir`：`tadd..tmin` 单 OP 能被改写为 OP-LIB call。
  2. 开启 `--enable-op-fusion` 时，组内支持 OP 能正常物化并继续后续流程。
- **seed 自动实例化**
  1. 单 seed（覆盖多 op + 多 dtype）可生成 `tmul/tdiv/tadd/tsub/tmax/tmin` 的实例调用。
  2. `f16/f32` 两个 dtype 均能命中与实例化。
- **选择与回退**
  1. 多候选按 `cost -> priority -> variant_id` 稳定选择。
  2. 单 OP 无匹配时保留原 OP 并出现 warning。
  3. fusion 组内失配时整组回退并 warning（不发生部分组物化）。
- **开关与错误**
  1. 默认开启 lowering 且未传 `--op-lib-dir`：直接报错。
  2. `--enable-op-fusion` + `--disable-oplib-lowering`：直接报错。
- **回归**
  1. 现有 tile_fusion 测试保持通过（必要时更新检查项为新日志/新属性）。
  2. 新增 `tmax/tmin` 专项 lit 用例（含 seed 与 variant 两条路径）。

### Assumptions
- Stage-1 只做 post-view 实现，不在本阶段切到 pre-view。
- 只保证 `f16/f32`，其它 dtype 不在首批范围内。
- OP-LIB 具体算法体不要求真实最优实现，可用 fake 模板，只要接口和选择机制正确。
- `op-fusion` 继续复用现有链路，但其 OP-LIB 选择/实例化完全依赖新的 oplib-lowering 基础能力。
