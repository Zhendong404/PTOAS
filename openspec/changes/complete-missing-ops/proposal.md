# Proposal: 补全缺失的基础算子 (trem, trems, tprelu, tlrelu)

## 目标 (Goals)
补全 `pto` 方言中缺失的基础算子，确保前端模型导出后能够完整映射到 PTO IR。

## 现状 (Current State)
目前 `pto` 方言已支持大部分二元算子和激活函数，但 `trem` (取模)、`trems` (标量取模)、`tprelu` (参数化 ReLU) 和 `tlrelu` (Leaky ReLU) 尚未定义，导致相关算子无法生成代码。

## 预期行为 (Expected Behavior)
1. 在 `PTOOps.td` 中完成 Op 定义。
2. 在 `PTO.cpp` 中完成必要的 C++ 验证器编写。
3. 确保这些 Op 能够通过基础的 IR 解析和打印测试。

## 影响范围 (Scope)
- `include/PTO/IR/PTOOps.td`
- `lib/PTO/IR/PTO.cpp`
- `test/basic/` (新增测试用例)
