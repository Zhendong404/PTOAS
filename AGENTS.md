# PTOAS 项目指南

## 项目概述

PTOAS (PTO Assembler & Optimizer) 是一个基于 LLVM/MLIR (release/19.x) 的专用编译器工具链，专为 PTO Bytecode (Programming Tiling Operator Bytecode) 设计。

作为连接上层 AI 框架与底层 NPU/GPGPU/CPU 硬件的桥梁，`ptoas` 采用 Out-of-Tree 架构，提供完整的 C++ 与 Python 接口。

## 主要职责

1. **IR 解析与验证**：解析 `.pto` 输入文件，验证 PTO Dialect 操作的语义正确性
2. **编译优化 (Passes)**：执行针对达芬奇架构的特定优化，如算子融合、自动同步插入等
3. **代码生成 (Lowering)**：将 PTO IR 下降到 `EmitC` / `Linalg` Dialect，生成可调用 `pto-isa` C++ 库的代码
4. **Python 绑定**：提供无缝集成的 Python 模块，支持 PyPTO、TileLang、CuTile 等框架

## 目录结构

```
pto-project/
├── include/PTO/           # PTO Dialect 头文件与 TableGen 定义
│   ├── IR/               # PTO Ops, Types, Attributes 定义
│   └── Transforms/       # Pass 头文件
├── lib/PTO/              # Dialect 核心实现
│   ├── IR/               # PTO Dialect 实现
│   └── Transforms/       # Pass 实现
│       └── InsertSync/   # 自动同步插入相关实现
├── lib/CAPI/             # C 语言接口
├── lib/Bindings/Python/  # Python Binding 实现
├── python/               # Python 模块
├── test/                 # 测试用例
│   ├── samples/          # 示例代码
│   └── npu_validation/   # NPU 验证脚本
├── tools/ptoas/          # 命令行工具入口
└── docs/                 # 文档
```

## 关键组件

### PTO Dialect

PTO Dialect 是项目的核心，定义了瓦片操作的操作集、类型和属性。

**主要操作类别**：
- **数据移动**：`pto.tload`, `pto.tstore`, `pto.tmov`
- **计算操作**：`pto.tmatmul`, `pto.tmatmul.bias`, `pto.tgemv`
- **内存管理**：`pto.alloc_tile`, `pto.bind_tile`, `pto.subset`
- **同步原语**：`pto.set_flag`, `pto.wait_flag`, `pto.barrier`
- **视图操作**：`pto.make_tensor_view`, `pto.partition_view`

**类型系统**：
- `!pto.tile_buf`：瓦片缓冲区类型
- `!pto.tensor_view`：张量视图类型
- `!pto.partition_tensor_view`：分区张量视图类型
- `!pto.ptr`：指针类型

### InsertSync 框架

InsertSync 是自动同步插入的核心框架，分析数据依赖并在必要时插入同步指令。

**主要阶段**：
1. **PTOIRTranslator**：将 MLIR 转换为内部 SyncIR 表示
2. **MemoryDependentAnalyzer**：分析内存依赖关系
3. **InsertSyncAnalysis**：基于依赖分析插入同步
4. **MoveSyncState**：优化同步位置（移出循环）
5. **RemoveRedundantSync**：移除冗余同步
6. **SyncEventIdAllocation**：分配事件 ID
7. **SyncCodegen**：生成最终的同步操作

### 其他 Passes

- **PTOPlanMemory**：内存规划和优化
- **InferPTOLayout**：布局推断
- **InferPTOMemScope**：内存作用域推断
- **PTOToEmitC**：降级到 EmitC
- **PTOConvertToDPS**：转换到 DPS 风格

### PTOAS Pass Pipeline 结构

以下基于 `tools/ptoas/ptoas.cpp` 当前实现整理。

**当前主干 Pipeline（默认）**

1. `func::FuncOp`：`LoweringSyncToPipe`
2. `ModuleOp`：`PTOViewToMemref`
3. `func::FuncOp`（可选）：`InferPTOLayout`（当未设置 `--disable-infer-layout`）
4. `ModuleOp`（可选）：`PlanMemory`（当 `--pto-level != level3`）
5. `func::FuncOp`（可选）：`PTOInsertSync`（当 `--enable-insert-sync` 且 `--pto-level != level3`）
6. `func::FuncOp`：`PTOMemrefToTileBuf`
7. `func::FuncOp`：`PTOCreateFusionGroupsPass`
8. `ModuleOp`：`PTOOutlineFusionGroupsPass`
9. `ModuleOp`：`PTOInstantiateAndLowerToLibCallPass`
10. `ModuleOp`：`PTOInlineLibCallPass`
11. `ModuleOp`：`PTOTileBufToMemref`
12. `ModuleOp`：`Canonicalizer -> CSE -> PTOLowLevelLoopFusionPass -> Canonicalizer -> CSE`
13. `ModuleOp`：`CSE -> EmitPTOManual -> emitc::FormExpressions -> CSE`
14. C++ 输出后处理（非 pass）：marker 重写与文本清理

**构建 Level 约束**

- `level1/level2`（默认 `level2`）：启用 `PlanMemory`，可选启用 `InsertSync`
- `level3`：跳过 `PlanMemory`，并忽略 `--enable-insert-sync`；要求 `alloc_tile` 显式提供 `addr`
- OP-Lib 主链路默认总是执行，不再由 `--enable-op-fusion` gating（该开关保留为兼容 no-op）
- `--op-lib-dir` 为必填，缺失直接报错
- `dump-ir-after-oplib-lowering` 在 `PTOInstantiateAndLowerToLibCallPass` 后截断
- `dump-ir-after-op-fusion` 在 `InlineLibCall + LoopFusion + TileBuf2Memref` 后截断

**OP Fusion V1 已落地约束**

1. 固定顺序：`InsertSync -> Memref2Tilebuf -> CreateFusionGroups -> OutlineFusionGroups -> InstantiateAndLowerToLibCall -> InlineLibCall -> Tilebuf2Memref`
2. OP-Lib 模板函数 / 实例函数 / 调用点接口必须是 `!pto.tile_buf`
3. Lower 阶段不允许把 OP-Lib 外部签名降到 `memref`
4. Inline 阶段允许在函数体内部临时插入 `tile_buf <-> memref` cast 以驱动低层 loop 生成
5. `PTOLowLevelLoopFusionPass` 默认总是执行（除非在更早 dump 截断）

说明：OP Fusion 的实现与约束见 `docs/tile_fusion/tile_fusion_plan.md` 与 `docs/tile_fusion/oplib_ir_spec.md`。

## 构建系统

项目使用 CMake 构建系统，依赖 LLVM/MLIR release/19.x。

**关键依赖**：
- LLVM/MLIR release/19.x
- CMake >= 3.20
- Ninja
- Python 3.8+
- pybind11

**构建步骤**：
1. 构建 LLVM/MLIR（共享库模式）
2. 配置 PTOAS（指向 LLVM 构建目录）
3. 编译并安装

详细构建说明见 `README.md`。

## 使用示例

### 命令行工具

```bash
# 解析并打印 PTO IR
ptoas tests/input.pto

# 运行自动同步插入并生成 C++ 代码
ptoas tests/input.pto --enable-insert-sync -o outputfile.cpp
```

### Python API

```python
from mlir.ir import Context, Module, Location
from mlir.dialects import pto

with Context() as ctx, Location.unknown():
    pto.register_dialect(ctx)
    module = Module.create()
    # 构建 PTO IR...
```

## 测试

项目包含多个测试类别：

- **Unit Tests**：基本功能测试（`test/basic/`）
- **Sample Tests**：示例代码测试（`test/samples/`）
- **NPU Validation**：NPU 上板验证（`test/npu_validation/`）

运行测试：

```bash
# 运行示例测试
cd test/samples/MatMul/
python3 tmatmulk.py > tmatmulk.pto
ptoas tmatmulk.pto -o tmatmulk.cpp

# NPU 验证
python3 test/npu_validation/scripts/generate_testcase.py \
  --input test/samples/Abs/abs-pto.cpp \
  --run-mode npu \
  --soc-version Ascend910B1
```

## 常见问题

**Q: 构建时出现 LLVM 版本错误？**
A: 确保使用 LLVM release/19.x 分支，并正确设置 LLVM_DIR 和 MLIR_DIR。

**Q: Python 绑定无法导入？**
A: 确保 PYTHONPATH 包含 MLIR 和 PTO 的 Python 路径，以及动态库路径。

**Q: 自动同步插入导致性能下降？**
A: 检查同步分析模式（NORMALSYNC vs BLOCKSYNC），并考虑手动优化关键路径。

## 相关文档

- `README.md`：详细构建和使用说明
- `docs/PTO_IR_manual.md`：PTO IR 规范手册
- `docs/tile_fusion/tile_fusion_plan.md`：OP Fusion V1 设计与落地方案（当前实现）
- `docs/tile_fusion/oplib_ir_spec.md`：OP-Lib 模板接口规范（tile_buf 约束）
- `PTO_OPS_SPEC.md`：PTO 操作规范
- `ReleaseNotes.md`：发布说明

## 贡献指南

1. 遵循项目的编码风格（LLVM/MLIR 风格）
2. 为新功能添加单元测试
3. 更新相关文档
4. 确保通过所有现有测试

## 许可证

本项目基于 Apache License 2.0 with LLVM Exceptions 发布。

---

*最后更新：2026-03-07*
