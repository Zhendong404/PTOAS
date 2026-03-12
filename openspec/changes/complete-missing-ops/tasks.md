# 任务列表: 补全缺失的基础算子

## 阶段 1: IR 定义 (ODS)
- [ ] 在 `include/PTO/IR/PTOOps.td` 中定义 `RemOp` (trem)。
- [ ] 在 `include/PTO/IR/PTOOps.td` 中定义 `RemSOp` (trems)。
- [ ] 在 `include/PTO/IR/PTOOps.td` 中定义 `PReluOp` (tprelu)。
- [ ] 在 `include/PTO/IR/PTOOps.td` 中定义 `LeakyReluOp` (tlrelu)。

## 阶段 2: C++ 验证逻辑实现
- [ ] 在 `lib/PTO/IR/PTO.cpp` 中实现 `RemOp` 和 `RemSOp` 的形状与类型校验。
- [ ] 在 `lib/PTO/IR/PTO.cpp` 中实现 `PReluOp` 的广播规则验证（权重与特征图通道一致性）。
- [ ] 在 `lib/PTO/IR/PTO.cpp` 中实现 `LeakyReluOp` 的 `alpha` 属性校验。

## 阶段 3: Codegen 支持
- [ ] 在 `lib/PTO/Transforms/PTOToEmitC.cpp` 中添加上述新算子的基础 EmitC 转换逻辑。

## 阶段 4: 测试与验证
- [ ] 创建 `test/basic/missing_ops.mlir` 回归测试文件。
- [ ] 运行 `python3 -m lit test/basic/missing_ops.mlir -v` 验证 IR 正确性。
- [ ] 确保算子能够正确打印和解析 (Roundtrip test)。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
