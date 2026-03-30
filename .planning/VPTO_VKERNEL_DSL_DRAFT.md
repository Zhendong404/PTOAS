# VPTO `pto.vkernel` Python DSL 草案

## 目标

定义一套面向 VPTO MLIR 的 Python DSL，入口装饰器为 `@pto.vkernel`。

该 DSL 的目标是：

- 在 Python 中描述 VPTO kernel
- 由 DSL 自己定义类型系统和操作语义
- 通过受限 Python 语法构造 DSL 抽象语法树
- 将 DSL 抽象语法树翻译为 VPTO MLIR

该方案当前仅作为设计草案持久化，接口与实现仍可继续调整。

## 非目标

当前草案不追求：

- 支持任意 Python 语法
- 直接复用 Python 运行时对象语义作为 IR 语义
- 第一阶段就覆盖全部 PTO/VPTO op
- 第一阶段就支持完整控制流和复杂推断

## 设计原则

- DSL 类型系统独立于 Python 原生类型系统
- Python 只作为承载语法，不直接决定 IR 语义
- 先支持静态、受限、可预测的 Python 子集
- `pto.vecscope` 和 `pto.strict_vecscope` 在 DSL 中是一等概念
- 先将 Python AST 翻译为 DSL IR，再由 DSL IR lowering 到 VPTO MLIR
- 文档中的主示例具有优先级；如果某个分项设计与主示例语义冲突，应以主示例语义表达为准回收设计

## 用户入口

推荐入口形式：

```python
import pto

@pto.vkernel
def abs_kernel(src: pto.ptr(pto.f32, "gm"),
               dst: pto.ptr(pto.f32, "gm")):
    pto.set_loop_size_outtoub(1, 1)

    ub_in = pto.castptr(0, pto.ptr(pto.f32, "ub"))
    ub_out = pto.castptr(4096, pto.ptr(pto.f32, "ub"))
    pto.copy_gm_to_ubuf(
        src, ub_in,
        0, 32, 128, 0, 0, False, 0, 128, 128,
    )

    pto.set_flag("PIPE_MTE2", "PIPE_V", "EVENT_ID0")
    pto.wait_flag("PIPE_MTE2", "PIPE_V", "EVENT_ID0")

    with pto.strict_vecscope(ub_in, ub_out, 0, 1024, 64, 1024) as (
        vin, vout, lb, ub, step, rem0
    ):
        rem: pto.i32 = rem0
        for lane in range(lb, ub, step):
            mask, rem = pto.plt_b32(rem)
            vec = pto.vlds(vin, lane)
            out = pto.vabs(vec, mask)
            pto.vsts(out, vout, lane, mask)

    pto.set_flag("PIPE_V", "PIPE_MTE3", "EVENT_ID0")
    pto.wait_flag("PIPE_V", "PIPE_MTE3", "EVENT_ID0")

    pto.set_loop_size_ubtoout(1, 1)
    pto.copy_ubuf_to_gm(ub_out, dst, 0, 32, 128, 0, 128, 128)
    pto.barrier("PIPE_ALL")
```

可选扩展形式：

```python
@pto.vkernel(name="abs_kernel", target="a5", verify=True)
def abs(src: pto.ptr(pto.f32, "gm"),
        dst: pto.ptr(pto.f32, "gm")):
    pass
```

建议装饰器返回一个描述对象，而不是普通 Python 可执行函数。该对象可提供：

- `mlir_module()`
- `mlir_text()`
- `verify()`
- `dump()`
- `emit(path)`

## 分层模型

建议实现分为三层：

### 1. DSL 前端层

负责接收 `@pto.vkernel` 标注的 Python 函数源码，并解析为 Python AST。

### 2. DSL 语义层

负责将 Python AST 翻译为 DSL IR。DSL IR 应显式表示：

- 类型
- 符号和值
- region / block
- capture 规则
- op schema
- 结构化控制流
- 统一的绑定环境与数据流

### 3. MLIR lowering 层

负责将 DSL IR 降低为 VPTO MLIR：

- DSL kernel -> `func.func`
- DSL vecscope -> `pto.vecscope`
- DSL strict vecscope -> `pto.strict_vecscope`
- DSL counted loop -> `scf.for`
- DSL if -> `scf.if`
- DSL ops -> `pto.*` / `arith.*`

## 类型系统

DSL 表面类型建议分为两层：

- Python 标量类型：
  - `bool`
  - `int`
  - `float`
- PTO 专有类型：
  - `pto.i1`
  - `pto.i8`
  - `pto.i16`
  - `pto.i32`
  - `pto.i64`
  - `pto.f16`
  - `pto.bf16`
  - `pto.f32`
  - `pto.mask`
  - `pto.align`

复合类型建议包含：

- `pto.ptr(elem_type, space)`
- `pto.vreg(lanes, elem_type)`

第一阶段建议仅支持：

- `space == "gm"`
- `space == "ub"`

示例：

```python
pto.ptr(pto.f32, "gm")
pto.ptr(pto.f32, "ub")
pto.vreg(64, pto.f32)
```

## 标量字面量与标量类型

标量字面量建议直接复用 Python 原生语法：

- `int`
- `float`
- `bool`

例如：

```python
0
1024
1.0
False
```

但 DSL 的标量语义类型仍应由 DSL 自己管理，而不是直接把 Python 运行时类型当作 IR 类型。

也就是说：

- Python `int` 只是字面量写法
- Python `float` 只是字面量写法
- Python `bool` 只是字面量写法
- 真正进入 DSL IR / VPTO MLIR 时，仍要落到 `i1`、`i32`、`i64`、`f32` 等后端标量类型

设计收敛：

- DSL 不向用户暴露 `index`
- `range(...)` 和 loop iv 的索引语义由 lowering 内部处理
- 用户层只区分“普通整型标量”和“循环索引位置上的内建语义”

### 强类型规则

对于会进入 IR 数据流边界的局部值，初版要求强类型，而不是依赖跨语句延迟决议。

这里的“IR 数据流边界”包括：

- `scf.for` 的 `iter_args`
- `scf.if` 的 merge 结果
- `pto.strict_vecscope` 的 region 参数
- 任何 block argument / region argument
- 任何需要显式结果类型的结构化结果

推荐允许的两种用户写法：

```python
remaining: pto.i32 = 1024
```

```python
remaining = pto.i32(1024)
```

其中：

- `remaining: pto.i32 = 1024` 表示用户通过 Python 注解语法显式声明机器整数位宽
- `pto.i32(1024)` 表示用户显式指定内部整型位宽

相对地，下面这种跨结构流动的裸字面量绑定不再作为推荐或保证语义：

```python
remaining = 1024
for lane in range(0, 1024, 64):
    mask, remaining = pto.plt_b32(remaining)
```

如果 `remaining` 需要成为 loop-carried state，则应先强类型化。

推荐规则：

- 语法层允许直接写 Python 标量字面量
- `bool` -> `i1`
- `float` -> 默认 `f32`
- `int` 不作为 DSL 机器整数类型的表面写法
- 固定宽度整数统一显式写成 `pto.i32` / `pto.i64`
- 若上下文不足以唯一确定类型，则报错或要求显式写类型

例如：

```python
ub_in = pto.castptr(0, pto.ptr(pto.f32, "ub"))
```

这里的 `0` 可以按 `i64` 常量解释；

```python
mask, rem = pto.plt_b32(1024)
```

这里的 `1024` 可以按 `i32` 常量解释；

```python
remaining = pto.i32(1024)
```

当用户需要明确位宽时，应允许并鼓励显式类型写法。

实现上，建议不要在各个 op builder 中分散处理字面量推断，而是统一通过 literal typing 阶段完成：

- AST `Constant`
- 未定型 DSL literal
- 由当前表达式上下文和 op schema 约束定型
- 生成 DSL 常量值 / MLIR constant

但这套弱延迟定型仅应用于“表达式级常量提升”：

- `tmp = 1024`
- `x = pid + tmp`

不应扩展到跨 `for` / `if` / region 的结构化数据流类型推断。

## 值模型

DSL 中的值不是 Python 原生值，而是符号值。

建议内部区分：

- `ConstValue`
- `OpResultValue`
- `BlockArgValue`

这些值只能参与 DSL 定义过的运算，不能作为普通 Python 真值直接参与运行时条件判断。

例如：

```python
x = pto.addi(a, b)
```

是合法 DSL 语义；

```python
if x:
    ...
```

不应被解释为普通 Python 语义。

对于 Python 原生标量字面量，也应先被提升为 DSL 常量值后再参与后续语义分析。

## 第一阶段支持的 Python 子集

建议第一阶段只支持：

- 顶层 kernel 函数定义
- 参数类型注解
- 赋值语句
- 带类型注解的赋值语句
- tuple unpack 赋值
- 表达式语句
- 空 `return`
- `with pto.vecscope(...)`
- `with pto.strict_vecscope(...) as (...)`
- `for i in range(lb, ub, step)`
- 整数 / 浮点 / 布尔常量

暂不支持：

- `while`
- `break` / `continue`
- 推导式
- 异常
- lambda
- 类
- 任意闭包自由捕获

## `pto.vecscope` DSL 语义

建议提供：

```python
with pto.vecscope():
    mask = pto.pset_b32("PAT_ALL")
    vec = pto.vlds(ub_in, lane)
    out = pto.vabs(vec, mask)
    pto.vsts(out, ub_out, lane, mask)
```

该形式表示一个普通 VPTO vector scope。

语义约束：

- body 可以引用外层 DSL 值
- 这种隐式引用不改变 `pto.vecscope` 自身的 region 接口形状
- lowering 到 `pto.vecscope`

## `pto.strict_vecscope` DSL 语义

建议提供：

```python
with pto.strict_vecscope(a, b, c) as (a_, b_, c_):
    mask = pto.pset_b32("PAT_ALL")
    vec = pto.vlds(a_, c_)
    out = pto.vabs(vec, mask)
    pto.vsts(out, b_, c_, mask)
```

语义约束：

- `a, b, c` 是显式 capture
- `a_, b_, c_` 是 region block arguments
- body 中不允许隐式捕获任何外层 DSL 值
- body 应通过 `a_, b_, c_` 访问所需外部输入
- body 只允许访问 `as (...)` 绑定名、body 内局部新定义名和字面量
- lowering 到 `pto.strict_vecscope`

这与当前 VPTO IR 中 `pto.strict_vecscope` 的接口语义保持一致。

二者的核心差异可以概括为：

- `pto.vecscope` 是开放 body，允许隐式读取外层绑定
- `pto.strict_vecscope` 是显式输入 body，只认 `as (...)` 绑定出的 region 输入

## 控制流语法

控制流建议按通用结构化语义定义，而不是为某个实现阶段单独收紧语义边界。

### `for`

建议 counted loop 直接使用 Python 原生 `range`：

```python
for lane in range(lb, ub, step):
    vec = pto.vlds(ub_in, lane)
    pto.vsts(vec, ub_out, lane, mask)
```

其 lowering 目标为 `scf.for`。

### `if`

条件控制流建议直接使用 Python `if`，其 lowering 目标为 `scf.if`。

对用户语义而言，`if` 结束后的绑定应等价于对应结构化分支合流后的结果：

- 两个分支都更新的外层绑定，通过统一的分支合流机制形成结构输出
- 仅被读取、不被更新的外层绑定，保持原绑定继续向后可见
- 仅在分支内部创建且未显式向外传播的局部绑定，不应泄漏到 `if` 外层

### 统一语义

`if`、`for`、`pto.vecscope`、`pto.strict_vecscope` 都应按统一的“绑定环境”模型解释：

- 结构进入时有一组当前可见绑定
- 结构执行过程中可产生新的绑定或更新已有绑定
- 结构结束时向外暴露一组更新后的当前绑定

lowering 层再将这种绑定环境变化映射为：

- `scf.if` 结果
- `scf.for` 的 `iter_args` / `scf.yield`
- region block arguments
- 普通 SSA 值（仅用于结构内部或无需跨结构传播的绑定）

在当前阶段，由于不支持 `break` / `continue`，循环控制流保持为标准 counted loop 语义，并统一 lowering 到 `scf.for`。

该模型不依赖针对某个具体语法片段的特判，而是依赖统一的数据流处理方式：

- `if` 使用统一的分支合流机制
- `for` 使用统一的迭代状态传递机制
- `vecscope` / `strict_vecscope` 使用统一的 region 输入绑定机制

文档当前不试图枚举所有 Python 场景的逐条处理细则，而是要求初版实现优先遵守这三种统一机制。

对于 loop-carried state，建议优先使用 Python 默认赋值语法来表达，而不是在用户层直接暴露 `yield` 风格 API。例如：

```python
rem: int = rem0
for lane in range(lb, ub, step):
    mask, rem = pto.plt_b32(rem)
    ...
```

DSL 语义层应将这种跨迭代保留并更新的绑定解释为 loop-carried state，并 lowering 为 `scf.for` 的 `iter_args` / `scf.yield`。

这里的 `rem` 在进入 loop-carried state 之前已经是强类型绑定；是否最终落成内部 `i32`，由 op schema 和 lowering 规则决定，但不再依赖跨结构的全局延迟推断。

对用户语义而言，loop 结束后的 `rem` 应等价于对应 `scf.for` 返回结果。
在 `@pto.vkernel` 中，这类重绑定按 DSL 的统一迭代状态传递机制解释，而不是按 Python 运行时普通局部变量覆盖语义解释。

对应地，`if` 内的绑定变化也应通过同一套绑定环境模型向结构外传播，而不是为 `if` 单独发明用户可见语法。

## 基础 op 草案

第一阶段建议优先支持这些 API：

- `pto.const(value, ty=None)`
- `pto.castptr(addr, ty)`
- `pto.addptr(ptr, offset)`
- `pto.set_loop_size_outtoub(...)`
- `pto.set_loop_size_ubtoout(...)`
- `pto.copy_gm_to_ubuf(...)`
- `pto.copy_ubuf_to_gm(...)`
- `pto.set_flag(...)`
- `pto.wait_flag(...)`
- `pto.barrier(...)`
- `pto.plt_b32(remaining)`
- `pto.pset_b32(pattern)`
- `pto.vlds(ptr, offset=None)`
- `pto.vsts(value, ptr, offset=None, mask=None)`
- `pto.vabs(vec, mask)`

后续逐步扩展：

- binary vector ops
- conversion ops
- reduction ops

## 建议的最小用户示例

```python
import pto

@pto.vkernel(target="a5")
def abs_kernel(src: pto.ptr(pto.f32, "gm"),
               dst: pto.ptr(pto.f32, "gm")):
    pto.set_loop_size_outtoub(1, 1)

    ub_in = pto.castptr(0, pto.ptr(pto.f32, "ub"))
    ub_out = pto.castptr(4096, pto.ptr(pto.f32, "ub"))
    pto.copy_gm_to_ubuf(
        src, ub_in,
        0, 32, 128, 0, 0, False, 0, 128, 128,
    )

    pto.set_flag("PIPE_MTE2", "PIPE_V", "EVENT_ID0")
    pto.wait_flag("PIPE_MTE2", "PIPE_V", "EVENT_ID0")

    with pto.strict_vecscope(ub_in, ub_out, 0, 1024, 64, 1024) as (
        vin, vout, lb, ub, step, rem0
    ):
        rem: pto.i32 = rem0
        for lane in range(lb, ub, step):
            mask, rem = pto.plt_b32(rem)
            vec = pto.vlds(vin, lane)
            out = pto.vabs(vec, mask)
            pto.vsts(out, vout, lane, mask)

    pto.set_flag("PIPE_V", "PIPE_MTE3", "EVENT_ID0")
    pto.wait_flag("PIPE_V", "PIPE_MTE3", "EVENT_ID0")

    pto.set_loop_size_ubtoout(1, 1)
    pto.copy_ubuf_to_gm(ub_out, dst, 0, 32, 128, 0, 128, 128)
    pto.barrier("PIPE_ALL")
```

期望生成的 IR 形态应接近：

```mlir
func.func @abs_kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>) {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %c64 = arith.constant 64 : index
  %c128_i64 = arith.constant 128 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c1024 = arith.constant 1024 : index
  %c1024_i32 = arith.constant 1024 : i32
  pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
  %0 = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
  %1 = pto.castptr %c4096_i64 : i64 -> !pto.ptr<f32, ub>
  %false = arith.constant false
  pto.copy_gm_to_ubuf %arg0, %0, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c0_i64, %false, %c0_i64, %c128_i64, %c128_i64
      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
  pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
  pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
  pto.strict_vecscope(%0, %1, %c0, %c1024, %c64, %c1024_i32) {
  ^bb0(%arg2: !pto.ptr<f32, ub>, %arg3: !pto.ptr<f32, ub>,
       %arg4: index, %arg5: index, %arg6: index, %arg7: i32):
    %_:1 = scf.for %arg8 = %arg4 to %arg5 step %arg6 iter_args(%arg9 = %arg7) -> (i32) {
      %mask, %next = pto.plt_b32 %arg9 : i32 -> !pto.mask, i32
      %vec = pto.vlds %arg2[%arg8] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
      %out = pto.vabs %vec, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
      pto.vsts %out, %arg3[%arg8], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
      scf.yield %next : i32
    }
  } : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, index, index, index, i32) -> ()
  pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
  pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
  pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
  pto.copy_ubuf_to_gm %1, %arg1, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c128_i64, %c128_i64
      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
  pto.barrier #pto.pipe<PIPE_ALL>
  return
}
```

该示例应与现有 `test/vpto/cases/abs/kernel.pto` 在核心计算语义上保持一致，只是通过 DSL 和 `pto.strict_vecscope` 的显式 capture 形式来表达 vector scope 边界。

## 错误模型

建议 DSL 在语义分析阶段尽量报 Python 源位置相关错误，例如：

- 不支持的 Python 语法
- 类型不匹配
- 非法隐式捕获
- 未定义符号
- 错误的 region 参数数量
- 非空 `return`

`pto.strict_vecscope` 的典型错误信息建议接近：

```text
strict_vecscope body cannot capture outer value 'ub_in' implicitly; pass it in the capture list and use the bound region argument instead
```

## 当前高风险点

当前方案的高风险点不在于某个具体场景缺少特判，而在于初版实现若偏离统一架构，容易重新退化为局部规则堆叠。

需要明确坚持的统一处理方式：

- `if` 统一走分支合流机制，不按具体代码形状逐条 special-case
- `for` 统一走迭代状态传递机制，不要求用户显式写 `yield`
- `vecscope` / `strict_vecscope` 统一走 region 输入绑定机制，不混入额外捕获策略
- Python 字面量统一通过一套类型推断入口进入 DSL IR，不按单个示例零散硬编码

初版实现允许覆盖面有限，但不应通过增加特判来替代这些统一机制。
当前阶段显式不支持 `break` / `continue`，以保持循环语义与 `scf.for` 的一致性。

## 分阶段实施建议

### 阶段 1

- `@pto.vkernel`
- 参数注解
- Python 原生标量字面量提升
- 基础类型系统
- tuple unpack 赋值
- 统一的分支合流机制
- 统一的迭代状态传递机制
- 统一的 region 输入绑定机制
- `pto.const`
- `pto.castptr`
- `pto.set_loop_size_outtoub`
- `pto.set_loop_size_ubtoout`
- `pto.copy_gm_to_ubuf`
- `pto.copy_ubuf_to_gm`
- `pto.set_flag`
- `pto.wait_flag`
- `pto.barrier`
- `pto.vlds`
- `pto.vsts`
- `pto.vabs`
- `pto.plt_b32`
- `pto.vecscope`
- `pto.strict_vecscope`
- Python `range`
- MLIR 文本输出

### 阶段 2

- 更多 VPTO op
- 更完整错误信息
- 索引与 pointer sugar

### 阶段 3

- 更丰富的类型推断
- 更丰富的语法糖
- 与现有 Python 绑定更自然地整合

## 当前建议

不建议第一步就尝试“任意 Python 语法 -> VPTO MLIR”。

更务实的方案是：

- 先做受限子集
- 先稳定 DSL IR 与类型系统
- 先把 `vecscope` / `strict_vecscope` 的 region 语义跑通
- 再逐步扩展控制流和语法糖
