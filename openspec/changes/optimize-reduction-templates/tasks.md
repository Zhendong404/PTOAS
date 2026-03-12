# 任务列表: Reduction 模板优化 (Seed 方案)

## 阶段 1: 模板重新设计
- [ ] 简化 `oplib/level3/reduction_templates.mlir`，提取通用的归约框架。
- [ ] 移除冗余的特定类型归约实例。

## 阶段 2: 种子系统 (Seed System) 实现
- [ ] 在 `lib/PTO/Transforms/PTOInstantiateAndLowerToLibCallPass` 中引入种子注入机制。
- [ ] 实现种子函数的动态查找与替换。

## 阶段 3: 算子迁移
- [ ] 将 `ReduceSum`, `ReduceMax`, `ReduceMin` 等算子迁移到新的种子系统。

## 阶段 4: 性能与正确性验证
- [ ] 编写测试用例验证归约结果的正确性。
- [ ] 检查生成的 C++ 代码是否符合 Da Vinci 架构的归约指令要求。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
