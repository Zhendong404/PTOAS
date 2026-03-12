# Proposal: 重构 OpLib Lowering 逻辑并引入接口化机制

## 目标 (Goals)
消除 `PTOLowerToOpLibCalls.cpp` 中硬编码的算子匹配逻辑，提高系统的可扩展性。

## 现状 (Current State)
当前的 `PTOLowerToOpLibCalls` 通过大量的 `dyn_cast` 和 `if-else` 分支来识别 `pto` 算子并寻找对应的 OpLib 模板。这种方式在算子增多时会导致代码迅速膨胀且难以维护。

## 预期行为 (Expected Behavior)
1. 定义 `OpLibOpInterface` 接口。
2. 让 `pto` 算子通过实现该接口来声明其对应的 OpLib 模板信息。
3. 重构 Lowering Pass，使其仅依赖接口进行转换。

## 影响范围 (Scope)
- `include/PTO/IR/PTOInterfaces.td`
- `include/PTO/IR/PTOOps.td`
- `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp`
