# 任务列表: OpLib Lowering 重构（细化版）

## 0. 审阅收敛与范围冻结
- [x] 0.1 从 `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp` 的 `buildMatchRequest` 导出基线映射表：`op -> kind/opName/operand 角色/特殊匹配字段`。
- [x] 0.2 在 `design.md` 明确接口契约必须覆盖的字段：`scalarPos`、`cmpMode`、`isBinary`、`requiredVariantId`、可选 `tmp` operand。
- [x] 0.3 明确本次重构边界：只替换算子匹配入口，不改 `TemplateRegistry` 选择语义和现有 `pto.oplib.*` attribute schema。
- [x] 0.4 定义完成标准：所有当前可 lowering 的 Op 都可以经接口路径组装 `MatchRequest`。

## 1. 接口契约落地（ODS + C++）
- [x] 1.1 在 `include/PTO/IR/PTOInterfaces.td` 新增 `OpLibOpInterface`，接口返回值可完整表达 `MatchRequest` 组装所需信息。
- [x] 1.2 在 `design.md` 记录接口 ABI 选择（方法签名、可选字段表达方式、错误处理约定）及其与现有 Pass 的兼容方案。
- [x] 1.3 完成接口相关 TableGen 产物接线并通过编译：`ninja -C build ptoas`。
- [x] 1.4 在 `lib/PTO/IR/PTO.cpp` 增加可复用 helper（例如 dtype/cmpMode 字符串转换），避免算子实现重复代码。

## 2. 算子侧分批接入接口
- [x] 2.1 批次 A（无特殊匹配字段）：`TAdd/TSub/TMul/TDiv/TMax/TMin/TAbs/TNeg/TRecip/TRelu/TExp/TLog/TSqrt/TRsqrt/TNot`。
- [x] 2.2 批次 B（tile-scalar/三元）：`TAddS/TSubS/TMulS/TDivS/TMaxS/TMinS/TAddC/TSubC/TAddSC/TSubSC/TLRelu/TPRelu/TRem/TRemS`，覆盖 `scalarPos` 与 `tdivs order` 约束。
- [x] 2.3 批次 C（比较/选择/位运算）：`TCmp/TCmpS/TSel/TSelS/TAnd/TOr/TXor/TShl/TShr/TAndS/TOrS/TXorS/TShlS/TShrS`，覆盖 `cmpMode/selectMode` 相关匹配。
- [x] 2.4 批次 D（reduction/broadcast）：`TRowSum/TRowMax/TRowMin/TColSum/TColMax/TColMin/TRowExpand/TColExpand/TRowExpandMul/TRowExpandDiv/TRowExpandSub/TExpands`，覆盖 `tmp` 和 `isBinary` 匹配。
- [x] 2.5 每个批次完成后执行最小增量构建 `ninja -C build ptoas`，避免累积式回归。

## 3. Pass 重构（双轨迁移到收敛）
- [x] 3.1 在 `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp` 新增接口驱动的 `buildMatchRequestFromInterface` 路径。
- [x] 3.2 迁移期保留现有 legacy `buildMatchRequest` 分支，仅在 Op 未实现接口时 fallback。
- [x] 3.3 增加调试日志区分 `interface` 与 `legacy` 路径，便于统计迁移覆盖率。
- [x] 3.4 批次 A-D 全部完成后删除 legacy `dyn_cast` 大分支并清理无用 helper/常量。
- [x] 3.5 确认 fusion group 与 single-op 两条 lowering 路径都复用同一接口入口，避免行为分叉。

## 4. 回归测试矩阵（按风险分层）
- [x] 4.1 基线回归：`test/oplib/oplib_single_op_lowering.mlir`、`test/oplib/missing_ops_oplib.mlir`。
- [x] 4.2 比较/位运算回归：`test/oplib/oplib_compare_select_family_ir.mlir`、`test/oplib/oplib_compare_bitwise_family_emitc.mlir`、`test/oplib/oplib_bitwise_shift_family_shapes.mlir`。
- [x] 4.3 reduction/broadcast 回归：`test/oplib/oplib_reduction_seed_variant.mlir`、`test/oplib/oplib_reduction_broadcast_family_shapes.mlir`、`test/oplib/oplib_reduction_broadcast_family_emitc.mlir`。
- [x] 4.4 兼容性回归：`test/oplib/oplib_rem_cmp_fallback.mlir`（验证 unsupported 场景仍保持既有告警/回退行为）。
- [x] 4.5 执行命令：`PTOAS_BIN=$PWD/build/tools/ptoas/ptoas llvm-lit -sv <上述用例>`；时间允许时补跑 `PTOAS_BIN=$PWD/build/tools/ptoas/ptoas llvm-lit -sv test/oplib`。

## 5. 文档与 OpenSpec 收口
- [x] 5.1 更新 `design.md` 接口章节，补充字段语义、迁移顺序、legacy 移除条件。
- [x] 5.2 如接口字段超出当前 requirement，补充 `specs/oplib-lowering/spec.md` 中的 scenario 文本使其可验收。
- [x] 5.3 执行并记录 `openspec validate refactor-oplib-lowering` 与 `openspec show refactor-oplib-lowering` 的结果。
- [x] 5.4 提交前核对未误改生成物：`build/`、`install/`、`build/output/`、`run.log`。

## 6. 4.5~4.9 强制 lowering 门禁（新增）
- [x] 6.1 在 `design.md`/`spec.md` 明确新契约：4.5~4.9 目标 op 必须命中 OP-Lib lowering，禁止静默 fallback。
- [x] 6.2 在 `PTOLowerToOpLibCalls.cpp` 将目标 op 的 fallback warning 升级为 hard error（匹配/选型/实例化/重写任一失败即 pass fail）。
- [x] 6.3 增加 pass 末尾残留检查：非 OP-Lib 用户函数中若仍有目标 op，直接报错。
- [x] 6.4 补齐模板/匹配缺口并清零 `build/output` / `build/output_log` 中 4.5~4.9 残留 failure（dtype、mask 类型、layout、缺失 family 模板）。
- [x] 6.4.1 基线审计：从 `build/output_log` 导出未命中 OP-Lib 的清单（op/kind/dtype/样例路径），并写入 `design.md` 的缺口矩阵。
- [x] 6.4.2 `dtype` 第一批补齐：`tand/tor/txor/tnot/tands/tors/txors` 的 `i16` 变体 + `tadd/tsub` 的 `f16` 变体（模板与匹配属性同步）。
- [x] 6.4.3 row-reduce/broadcast 布局补齐：支持 `col_major 16x1` 行向量路径（`trowmax/trowsum/trowexpand*`）。
- [x] 6.4.4 compare/select 非同构 operand 收敛：统一 `tcmp/tcmps/tsel` mask 类型策略并补 verifier + 模板 + 回归。
- [x] 6.4.5 回归收口：`llvm-lit -sv test/oplib` + 关键样例复跑，确认不再出现 4.5~4.9 `no matching OP-Lib entry`。
- [x] 6.5 更新回归用例：将 `test/oplib/oplib_rem_cmp_fallback.mlir` 改为“必须失败”语义，验证 strict gate 生效。
