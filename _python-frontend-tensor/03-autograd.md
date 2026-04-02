# 03. Autograd 自动微分系统

> 本文档深入解析 PyTorch 自动微分系统的实现原理，包括计算图构建、反向传播引擎和梯度管理模式

---

## 01. Autograd 概述

### 1.1 Autograd 的位置

Autograd 是 PyTorch 的自动微分引擎，位于 Python 前端与 C++ 核心之间：

```
用户代码 (Tensor 操作)
    ↓
┌─────────────────────────────────────┐
│  Python 层 (torch/autograd/)        │
│  - grad_mode.py (梯度模式)          │
│  - function.py (Function 基类)       │
│  - variable.py (Variable 包装)       │
│  - gradcheck.py (梯度检查)          │
├─────────────────────────────────────┤
│  C++ 层 (torch/csrc/autograd/)      │
│  - engine.cpp (反向传播引擎)         │
│  - variable.h (Variable C++ 定义)    │
│  - function.h (Function C++ 定义)    │
│  - graph_task.h (计算图任务)         │
└─────────────────────────────────────┘
    ↓
ATen 算子执行
```

### 1.2 核心概念

| 概念 | 说明 | 源码位置 |
|------|------|----------|
| `Tensor` | 带自动微分能力的张量 | `torch/_tensor.py` |
| `Variable` | Tensor 的包装类（历史遗留） | `torch/autograd/variable.py` |
| `Function` | 自动微分函数基类 | `torch/autograd/function.py` |
| `grad_fn` | 创建 Tensor 的函数引用 | C++ 属性 |
| `GraphTask` | 一次反向传播任务 | `torch/csrc/autograd/graph_task.h` |
| `Engine` | 反向传播引擎 | `torch/csrc/autograd/engine.cpp` |

---

## 02. 梯度管理模式

### 2.1 is_grad_enabled 全局状态

**源码位置**: `torch/__init__.py` (C++ 绑定)

```python
# 全局梯度计算开关
torch.is_grad_enabled()  # bool
torch.set_grad_enabled(True/False)  # None
```

### 2.2 no_grad 上下文管理器

**源码位置**: `torch/autograd/grad_mode.py:21-86`

```python
class no_grad(_NoParamDecoratorContextManager):
    r"""
    禁用梯度计算的上下文管理器
    
    适用于推理阶段，减少内存消耗
    
    Example::
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     y = x * 2  # y.requires_grad = False
        >>> y.requires_grad
        False
    """
    
    def __init__(self) -> None:
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self.prev = False
    
    def __enter__(self) -> None:
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
    
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.set_grad_enabled(self.prev)
```

**实现细节**:
- 使用 `_NoParamDecoratorContextManager` 基类
- 同时支持上下文管理器 (`with`) 和装饰器 (`@torch.no_grad()`)
- 线程局部状态，不影响其他线程

### 2.3 enable_grad 上下文管理器

**源码位置**: `torch/autograd/grad_mode.py:88-141`

```python
class enable_grad(_NoParamDecoratorContextManager):
    r"""
    启用梯度计算的上下文管理器
    
    用于在 no_grad 环境中临时启用梯度计算
    
    Example::
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     with torch.enable_grad():
        ...         y = x * 2  # y.requires_grad = True
        >>> y.requires_grad
        True
    """
    
    def __enter__(self) -> None:
        self.prev = torch.is_grad_enabled()
        torch._C._set_grad_enabled(True)
    
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_grad_enabled(self.prev)
```

### 2.4 set_grad_enabled 上下文管理器

**源码位置**: `torch/autograd/grad_mode.py:143-200`

```python
class set_grad_enabled(_DecoratorContextManager):
    r"""
    根据参数设置梯度计算开关
    
    Args:
        mode (bool): True 启用，False 禁用
    
    Example::
        >>> is_train = False
        >>> with torch.set_grad_enabled(is_train):
        ...     y = x * 2
    """
    
    def __init__(self, mode: bool) -> None:
        self.prev = torch.is_grad_enabled()
        self.mode = mode
        torch._C._set_grad_enabled(mode)
    
    def __enter__(self) -> None:
        torch._C._set_grad_enabled(self.mode)
    
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_grad_enabled(self.prev)
```

### 2.5 inference_mode 上下文管理器

**源码位置**: C++ 实现 (`torch/csrc/autograd/InferenceMode.h`)

```python
class inference_mode:
    r"""
    推理模式 - 比 no_grad 更严格的梯度禁用模式
    
    与 no_grad 的区别:
    - 不仅禁用梯度，还禁用 Autograd 引擎的其他开销
    - 允许更多优化（如原地操作检查放宽）
    - 性能略优于 no_grad
    
    适用场景：纯推理，不需要任何梯度相关功能
    """
    pass

# 使用示例
with torch.inference_mode():
    output = model(input)
```

---

## 03. Function 类与自定义 Autograd

### 3.1 Function 基类

**源码位置**: `torch/autograd/function.py`

```python
class Function(torch.autograd.function._FuncBase):
    r"""
    自动微分函数基类
    
    所有 autograd 函数都继承自 Function
    
    子类必须实现:
    - forward(ctx, ...) -> Tensor: 前向计算
    - backward(ctx, grad_output) -> Tuple[Tensor, ...]: 反向梯度计算
    
    Example::
    
        class Exp(Function):
            @staticmethod
            def forward(ctx, i):
                result = i.exp()
                ctx.save_for_backward(result)  # 保存用于 backward 的张量
                return result
            
            @staticmethod
            def backward(ctx, grad_output):
                result, = ctx.saved_tensors
                return grad_output * result  # d(exp(x))/dx = exp(x)
    """
    
    _is_backward_hook = False
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._forward_from_backward_fn = None
```

### 3.2 自定义 Function 示例

```python
import torch

class MyLinearFunction(torch.autograd.Function):
    """自定义线性层 Autograd 函数"""
    
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # 保存反向传播需要的张量
        ctx.save_for_backward(input, weight, bias)
        
        # 前向计算
        output = input @ weight.t()
        if bias is not None:
            output += bias
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的张量
        input, weight, bias = ctx.saved_tensors
        
        # 计算梯度
        grad_input = grad_output @ weight
        grad_weight = grad_output.t() @ input
        grad_bias = grad_output.sum(0) if bias is not None else None
        
        # 返回每个输入的梯度 (None 表示该输入不需要梯度)
        return grad_input, grad_weight, grad_bias


# 使用方式
def linear(input, weight, bias):
    return MyLinearFunction.apply(input, weight, bias)
```

### 3.3 save_for_backward 与 saved_tensors

**源码位置**: `torch/autograd/function.py`

```python
class Function:
    @classmethod
    def save_for_backward(cls, *tensors):
        """
        保存张量供 backward 使用
        
        保存的张量存储在 ctx.saved_tensors 中
        
        注意:
        - 只能保存 requires_grad=True 或与计算图相关的张量
        - 保存的张量会阻止其存储被释放
        """
        pass
    
    @property
    def saved_tensors(self):
        """
        获取保存的张量元组
        
        如果只需要其中一个，使用 saved_tensors[0] 等
        """
        pass
    
    @property
    def saved_variables(self):
        """
        (已弃用) 获取保存的 Variable 列表
        
        现代代码应使用 saved_tensors
        """
        pass
```

### 3.4 mark_dirty 与 mark_non_differentiable

```python
class Function:
    @classmethod
    def mark_dirty(cls, ctx, *tensors):
        """
        标记原地修改的张量
        
        如果 forward 中原地修改了某个输入，必须标记为 dirty
        """
        pass
    
    @classmethod
    def mark_non_differentiable(cls, ctx, *tensors):
        """
        标记不需要微分的输出
        
        适用于输出中包含不需要梯度的张量（如索引）
        """
        pass
```

### 3.5 NestedIOFunction

```python
class NestedIOFunction(Function):
    """
    支持嵌套 I/O 的 Function 基类
    
    用于处理输入/输出为嵌套结构的场景
    """
    pass
```

---

## 04. backward 函数与反向传播引擎

### 4.1 torch.autograd.backward

**源码位置**: `torch/autograd/__init__.py`

```python
def backward(
    tensors: Union[Sequence[_TensorOrTensorsOrGradEdge], _TensorOrTensorsOrGradEdge],
    grad_tensors: Optional[Union[Sequence[_TensorOrOptionalTensors], _TensorOrOptionalTensors]] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    grad_variables: Optional[Union[Sequence[_TensorOrOptionalTensors], _TensorOrOptionalTensors]] = None,
    inputs: Optional[Sequence[torch.Tensor]] = None,
) -> None:
    r"""
    计算并累积梯度
    
    计算图从 given tensors 开始，反向传播到所有 requires_grad=True 的叶子节点
    
    Args:
        tensors: 要微分的张量（通常是标量损失）
        grad_tensors: 对应张量的梯度（雅可比向量积）
        retain_graph: 是否保留计算图
        create_graph: 是否构建导数图（用于高阶导数）
        inputs: 指定要累积梯度的叶子节点
    
    Example::
    
        >>> x = torch.randn(3, requires_grad=True)
        >>> y = x * 2
        >>> while y.data.norm() < 1000:
        ...     y = y * 2
        >>> y.backward()
    """
    # 参数处理...
    
    # 调用 C++ 引擎
    _engine_run_backward(
        tensors,
        grad_tensors_,
        retain_graph,
        create_graph,
        inputs,
        allow_unreachable=True,
        accumulate_grad=True,
    )
```

### 4.2 _engine_run_backward

**源码位置**: `torch/autograd/graph.py`

```python
def _engine_run_backward(*args, **kwargs):
    """
    运行 C++ 反向传播引擎
    
    实际引擎实现在 torch/csrc/autograd/engine.cpp
    """
    pass
```

### 4.3 C++ Engine 引擎

**源码位置**: `torch/csrc/autograd/engine.cpp`

```cpp
// Engine 类定义（简化版）
class Engine {
public:
    // 主反向传播函数
    std::vector<Variable> backward(
        const edge_list& roots,
        const variable_list& inputs,
        bool retain_graph,
        bool create_graph
    );
    
    // 单节点反向传播
    void execute(
        edge_list& roots,
        variable_list& grads,
        bool retain_graph,
        bool create_graph
    );
    
    // 获取单例
    static Engine& get_default_engine();
    
private:
    // 计算图调度
    void compute_sequence_numbers(...);
    
    // 节点执行
    void evaluate_function(...);
};
```

**反向传播流程**:

```
1. 构建计算图 (从 roots 向后遍历)
       ↓
2. 拓扑排序 (确定执行顺序)
       ↓
3. 计算每个节点的"入度"(依赖数)
       ↓
4. 使用队列执行反向传播:
   - 将入度为 0 的节点加入就绪队列
   - 执行节点的 grad_fn
   - 将梯度传递给前驱节点
   - 减少前驱节点的入度
   - 重复直到所有节点处理完毕
       ↓
5. 累积梯度到叶子节点的 .grad
```

---

## 05. 计算图与 grad_fn

### 5.1 grad_fn 属性

每个由操作创建的 Tensor 都有一个 `grad_fn` 属性：

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2

print(y.grad_fn)
# 输出：<MulBackward0 object at 0x...>

# grad_fn 是指向 C++ FunctionNode 的引用
print(type(y.grad_fn))
# 输出：<class 'torch.autograd.function.MulBackward0'>
```

### 5.2 计算图结构

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
z = y ** 2
w = z ** 2

w.backward()

# 计算图:
# w (grad_fn=PowBackward0)
#   ↑
# z (grad_fn=PowBackward0)
#   ↑
# y (grad_fn=PowBackward0)
#   ↑
# x (leaf, requires_grad=True)
```

### 5.3 叶子节点

```python
x = torch.randn(3, requires_grad=True)  # 叶子节点
y = x + 1  # 非叶子节点

print(x.is_leaf)   # True
print(y.is_leaf)   # False
print(x.grad_fn)   # None (叶子节点没有 grad_fn)
print(y.grad_fn)   # <AddBackward0>
```

### 5.4 retain_graph 参数

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
z = y ** 2

# 第一次反向传播
z.backward()  # 默认 retain_graph=False，计算图被释放

# 再次反向传播会出错
# z.backward()  # RuntimeError: Trying to backward through the graph a second time

# 解决方案：设置 retain_graph=True
z = y ** 2
z.backward(retain_graph=True)  # 保留计算图
z.backward()  # 可以再次调用
```

---

## 06. Variable 类（历史）

### 6.1 Variable 与 Tensor 的合并

在 PyTorch 0.4.0 之前，`Variable` 是独立于 `Tensor` 的包装类。0.4.0 之后两者合并。

**源码位置**: `torch/autograd/variable.py`

```python
# 现在的 Variable 只是 Tensor 的别名
Variable = torch.Tensor

# 历史代码中可能看到:
from torch.autograd import Variable
x = Variable(torch.randn(3, requires_grad=True))

# 现代代码:
x = torch.randn(3, requires_grad=True)
```

### 6.2 遗留 API

```python
# 已弃用，但可能仍在使用
Variable(data, requires_grad=True)  # 等价于 Tensor(data, requires_grad=True)
```

---

## 07. 梯度检查工具

### 7.1 gradcheck

**源码位置**: `torch/autograd/gradcheck.py`

```python
def gradcheck(
    func,
    inputs,
    eps=1e-6,
    atol=1e-5,
    rtol=1e-5,
    nondet_tol=0.0,
):
    r"""
    使用有限差分法检查梯度计算是否正确
    
    Args:
        func: 要检查的函数
        inputs: 输入张量
        eps: 有限差分步长
        atol: 绝对误差容忍度
        rtol: 相对误差容忍度
    
    Returns:
        bool: 梯度是否正确
    
    Example::
    
        >>> import torch.autograd.gradcheck as gradcheck
        >>> def f(x):
        ...     return (x ** 2).sum()
        >>> x = torch.randn(3, requires_grad=True, dtype=torch.double)
        >>> gradcheck(f, x)
        True
    """
    pass
```

### 7.2 gradgradcheck

```python
def gradgradcheck(
    func,
    inputs,
    grad_outputs=None,
    eps=1e-6,
    atol=1e-5,
    rtol=1e-5,
):
    r"""
    检查高阶梯度（二阶导数）
    
    类似于 gradcheck，但检查的是梯度的梯度
    """
    pass
```

---

## 08. 异常检测模式

### 8.1 detect_anomaly

**源码位置**: `torch/autograd/anomaly_mode.py`

```python
def set_detect_anomaly(mode):
    """
    启用/禁用异常检测模式
    
    启用后，会检测:
    - 反向传播中的 NaN/Inf
    - 创建非法节点的函数
    """
    torch._C._set_detect_anomaly(mode)


def detect_anomaly():
    """
    异常检测上下文管理器
    
    Example::
    
        >>> with torch.autograd.detect_anomaly():
        ...     loss = compute_loss()
        ...     loss.backward()
        # 如果检测到 NaN，会抛出异常并显示创建该节点的代码位置
    """
    pass
```

---

## 09. forward_ad - 前向模式自动微分

### 9.1 前向模式 vs 反向模式

| 特性 | 反向模式 (默认) | 前向模式 |
|------|----------------|----------|
| 适用场景 | 多输入，单输出 (如损失函数) | 单输入，多输出 |
| 效率 | O(输出维度) | O(输入维度) |
| API | `Tensor.backward()` | `jvp()` |

### 9.2 jvp (Jacobian-Vector Product)

```python
from torch.autograd import forward_ad

# 计算 Jacobian-向量积
def jvp(func, primals, tangents):
    """
    前向模式自动微分
    
    Args:
        func: 函数
        primals: 原始输入
        tangents: 切向量
    
    Returns:
        (primals_out, jvp)
    """
    pass
```

---

## 附录：核心源码文件索引

### Python 层 (`torch/autograd/`)

| 文件 | 内容 |
|------|------|
| `__init__.py` | backward, gradcheck 等入口函数 |
| `function.py` | Function 基类 |
| `grad_mode.py` | no_grad, enable_grad 等 |
| `gradcheck.py` | 梯度检查 |
| `anomaly_mode.py` | 异常检测 |
| `variable.py` | Variable 别名 |
| `graph.py` | _engine_run_backward |

### C++ 层 (`torch/csrc/autograd/`)

| 文件 | 内容 |
|------|------|
| `engine.cpp` | 反向传播引擎 |
| `engine.h` | Engine 类定义 |
| `variable.h` | Variable C++ 定义 |
| `variable.cpp` | Variable C++ 实现 |
| `function.h` | Function C++ 定义 |
| `graph_task.h` | GraphTask 定义 |
| `python_function.cpp` | Python Function 绑定 |

---

## 后续章节

- [04. Storage 与内存管理](./04-storage-memory.md) - 底层存储与内存分配
- [05. 工厂函数实现](./05-factory-functions.md) - Tensor 创建机制
