# Design: OpLib Lowering 重构

## 详细设计 (Detailed Design)

### 1. 接口定义 (`OpLibOpInterface`)
在 `PTOInterfaces.td` 中定义：
- `getOpLibTemplateName()`: 返回字符串，指定对应的 `.mlir` 模板函数名。
- `getOpLibOperands()`: 返回操作数列表，允许 Op 自定义哪些参数传递给模板。

### 2. 算子适配
- 为主要算子实现接口。

### 3. Pass 逻辑重构
- 遍历函数中的所有 Op。
- 检查是否实现了 `OpLibOpInterface`。
- 如果实现，则获取模板名称和参数，调用 `OpLibInstantiator` 进行实例化。

## 交付物 (Deliverables)
- 接口定义文件。
- 重构后的算子定义。
- 简化后的 `PTOLowerToOpLibCalls.cpp`。
