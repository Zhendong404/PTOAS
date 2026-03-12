# 任务列表: OpLib Lowering 重构

## 阶段 1: 接口定义 (ODS)
- [ ] 在 `include/PTO/IR/PTOInterfaces.td` 中新增 `OpLibOpInterface`。
- [ ] 定义接口方法 `getOpLibTemplateName()`。
- [ ] 定义接口方法 `getOpLibOperands()`。

## 阶段 2: 算子适配
- [ ] 在 `include/PTO/IR/PTOOps.td` 中为常用的 Elementwise 算子添加 `OpLibOpInterface` 声明。
- [ ] 在 `lib/PTO/IR/PTO.cpp` 中为每个算子实现对应的接口方法。

## 阶段 3: Pass 逻辑重构
- [ ] 修改 `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp`，移除硬编码的算子匹配逻辑。
- [ ] 实现通用的接口检测与模板实例化调用流程。
- [ ] 清理冗余的匹配代码，提高扩展性。

## 阶段 4: 回归测试
- [ ] 运行现有的 OpLib lit 测试 (`test/oplib/*.mlir`)，确保重构后功能无损。
- [ ] 验证 A5 架构下的代码生成路径是否依然正确。

Co-Authored-By: Claude Opus 4.6 <noreply@anthreply.com>
