# Design: 补全缺失的基础算子

## 详细设计 (Detailed Design)

### 1. ODS 定义
- **`trem` (RemOp)**: 继承自 `BinaryTileOp`，支持两个 Tile 之间的按元素取模运算。
- **`trems` (RemSOp)**: 继承自 `BinaryTileScalarOp`，支持 Tile 与标量之间的取模。
- **`tprelu` (PReluOp)**: 输入为特征图 Tile 和权重 Tile（通常是 Channel 维度）。
- **`tlrelu` (LeakyReluOp)**: 输入为特征图 Tile 和一个浮点数属性 `alpha`。

### 2. 验证逻辑
- `trem`/`trems`: 验证输入 Tile 的形状和类型必须匹配。
- `tprelu`: 验证权重 Tile 的形状是否符合广播规则（通常要求与特征图的 C 维度一致）。
- `tlrelu`: 验证 `alpha` 属性是否存在。

### 3. 代码复用
- 尽可能复用已有的 Trait，如 `SameOperandsAndResultType` 或自定义的 `PTO_TileOpInterface`。

## 交付物 (Deliverables)
- 修改后的 `PTOOps.td` 和 `PTO.cpp`。
- 新增的 lit 测试文件 `test/basic/missing_ops.mlir`。
