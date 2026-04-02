# 06. Dispatcher 调度系统

> 本文档深入解析 PyTorch 的 Dispatch Key 调度机制与算子注册系统

---

## 01. Dispatcher 概述

### 1.1 什么是 Dispatch

Dispatch（调度）是 PyTorch 将算子调用路由到正确实现的核心机制。

```
用户调用：torch.add(a, b)
         ↓
    Dispatcher
         ↓
    检查 Dispatch Key 队列
         ↓
    [Autograd] → [XLA] → [CUDA] → [CPU] → [Composite]
         ↓
    选择正确的 Kernel 实现
         ↓
    执行并返回结果
```

### 1.2 Dispatch Key 层次结构

**源码位置**: `torch/_C/__init__.pyi.in`

```python
# DispatchKey 枚举（部分）
class DispatchKey(Enum):
    # 最高优先级：调试与分析
    Named           # 命名维度
    FuncTorchDynamicShapeBackwards  # 动态形状反向
    
    # 自动微分
    AutogradXLA     # XLA 自动微分
    AutogradCUDA    # CUDA 自动微分
    AutogradCPU     # CPU 自动微分
    AutogradNestedTensor  # 嵌套张量自动微分
    AutogradDispatchKey  # 通用自动微分
    
    # 设备特定后端
    XLA             # TPU
    CUDA            # NVIDIA GPU
    MPS             # Apple Silicon
    PrivateUse1     # 自定义后端
    IPU             # Graphcore IPU
    XPU             # Intel GPU
    
    # 特殊功能
    Quantized       # 量化算子
    SparseCsr       # 稀疏 CSR 格式
    Sparse          # 稀疏张量
    Mkldnn          # MKL-DNN
    GLSL            # OpenGL 着色语言
    
    # 复合实现
    CompositeExplicitAutograd  # 复合实现（带自动微分）
    CompositeExplicitAutogradNonFunctional
    CompositeExplicitDeviceDispatch
    
    # 最低优先级：回退
    BackendSelect   # 后端选择
    AutocastCPU     # 自动混合精度 CPU
    AutocastCUDA    # 自动混合精度 CUDA
    FuncTorchBatched  # 批量处理
    FuncTorchVmap   # 向量化映射
    
    # 特殊
    Meta            # 元张量（不分配内存）
    Zero            # 零张量
```

---

## 02. Dispatch Key 执行流程

### 2.1 完整的 Dispatch 流程

```
1. 用户调用 torch.add(a, b)
         ↓
2. 生成初始 Dispatch Key 队列
   [Autograd, DeviceSpecific, BackendSelect]
         ↓
3. Dispatcher 遍历队列
   for key in dispatch_keys:
       if has_kernel(key):
           execute_kernel(key)
           break
         ↓
4. 找到匹配的实现并执行
```

### 2.2 Dispatch Key 包含/排除机制

**源码位置**: `torch/_dispatch/python.py`

```python
# 启用/禁用特定 Dispatch Key
torch._C._EnableDispatchKey(key)
torch._C._DisableDispatchKey(key)

# 上下文管理器
with torch._C._IncludeDispatchKeyGuard(key):
    # 包含特定 key
    pass

with torch._C._ExcludeDispatchKeyGuard(key):
    # 排除特定 key
    pass
```

### 2.3 DispatchKeySet

**源码位置**: `torch/library.py:944-947`

```python
# 创建 DispatchKey 集合
autocast_keyset = torch._C.DispatchKeySet(
    torch._C.DispatchKey.AutocastCPU
) | torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastCUDA)

# 在排除特定 key 的情况下执行
with torch._C._ExcludeDispatchKeyGuard(autocast_keyset):
    result = op(*args, **kwargs)
```

---

## 03. 算子注册系统

### 03.1 torch.library 模块

**源码位置**: `torch/library.py`

`torch.library` 提供了注册自定义算子的 API。

```python
import torch
from torch.library import impl

# 方法 1: 使用装饰器注册
@impl("aten::my_op", "CPU")
def my_op_cpu(x):
    return x * 2

# 方法 2: 使用库对象
my_lib = torch.library.Library("aten", "IMPL")

@my_lib.impl("my_op", "CPU")
def my_op_cpu(x):
    return x * 2

# 方法 3: 直接注册
my_lib.impl("my_op", my_op_cpu, "CPU")
```

### 3.2 Library 类

**源码位置**: `torch/library.py:200-400`

```python
class Library:
    """
    算子注册库
    
    Args:
        namespace: 命名空间（如 "aten", "my_custom"）
        kind: 注册类型 ("DEF", "IMPL", "PRIVATEUSE1")
        _stacklevel: 堆栈级别（用于错误报告）
    """
    
    def __init__(self, namespace, kind, _stacklevel=2):
        self.namespace = namespace
        self.kind = kind
        self._stacklevel = _stacklevel
    
    def impl(
        self,
        name: str,
        fn: Callable,
        dispatch_key: str | DispatchKey,
        *,
        with_keyset: bool = False,
    ):
        """
        注册算子实现
        
        Args:
            name: 算子名称（如 "my_op.Tensor"）
            fn: 实现函数
            dispatch_key: DispatchKey（如 "CPU", "CUDA"）
            with_keyset: 是否将 DispatchKeySet 作为第一个参数传入
        
        Example::
            
            lib = torch.library.Library("aten", "IMPL")
            
            @lib.impl("my_op", "CPU")
            def my_op_cpu(x):
                return x * 2
        """
        # 实际注册逻辑在 C++ 层
        pass
    
    def define(self, schema: str):
        """
        定义新的算子（仅声明，不实现）
        
        Args:
            schema: 算子签名（类似函数声明）
        
        Example::
            
            lib = torch.library.Library("my_namespace", "DEF")
            lib.define("my_op(Tensor x) -> Tensor")
        """
        pass
    
    def register(self, name: str, fn: Callable, dispatch_key: str = None):
        """
        注册算子（define + impl 的便捷组合）
        """
        pass
```

### 3.3 算子签名（Schema）

```python
# Schema 格式
# op_name(arg1: Type1, arg2: Type2) -> ReturnType

# 示例
lib.define("add(Tensor self, Tensor other) -> Tensor")
lib.define("matmul(Tensor self, Tensor other) -> Tensor")
lib.define("sum(Tensor self, int[]? dim, bool keepdim) -> Tensor")

# 类型说明
# Tensor       - 张量
# Tensor?      - 可选张量
# int          - 整数
# float        - 浮点数
# bool         - 布尔值
# str          - 字符串
# int[]        - 整数列表
# Scalar       - 标量（int 或 float）
# Device       - 设备
# dtype        - 数据类型
```

---

## 04. Python Dispatcher

### 4.1 enable_python_dispatcher

**源码位置**: `torch/_dispatch/python.py:1-25`

```python
__all__ = [
    "enable_python_dispatcher",
    "no_python_dispatcher", 
    "enable_pre_dispatch"
]

no_python_dispatcher = torch._C._DisablePythonDispatcher
enable_python_dispatcher = torch._C._EnablePythonDispatcher
enable_pre_dispatch = torch._C._EnablePreDispatch


@contextmanager
def enable_python_dispatcher():
    """
    启用 Python 层的 Dispatcher
    
    允许在 Python 层拦截算子调用并自定义行为
    """
    with torch._C._EnablePythonDispatcher():
        yield
```

### 4.2 __torch_dispatch__ 协议

```python
class MyTensor(torch.Tensor):
    """自定义 Tensor 子类，拦截算子调用"""
    
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """
        拦截所有通过此 Tensor 类型的算子调用
        
        Args:
            func: 被调用的算子（OpOverload）
            types: 所有参数的类型
            args: 位置参数
            kwargs: 关键字参数
        
        Returns:
            算子的结果
        """
        print(f"Dispatching {func}")
        
        # 调用默认实现
        return super().__torch_dispatch__(func, types, args, kwargs)


# 使用示例
x = MyTensor([1.0, 2.0, 3.0])
y = x + 1  # 会打印 "Dispatching aten::add.Tensor"
```

### 4.3 跨引用功能化（Cross-reference Functionalization）

**源码位置**: `torch/_dispatch/python.py:120-160`

```python
def make_crossref_functionalize(
    op: torch._ops.OpOverload,
    final_key: DispatchKey
) -> Callable | DispatchKey:
    """
    创建跨引用功能化的实现
    
    用于验证不同 Dispatch Key 的实现是否产生一致的结果
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    
    def handler(*args, **kwargs):
        fake_mode = FakeTensorMode()
        
        # 将输入转换为 FakeTensor
        fake_args = tree_map(fake_mode.from_tensor, args)
        
        # 执行并比较结果
        result = op(*fake_args, **kwargs)
        
        return result
    
    return handler
```

---

## 05. 常见 Dispatch Key 详解

### 5.1 Autograd 相关 Key

```python
# AutogradCPU / AutogradCUDA
# 在执行实际算子前/后记录梯度计算图

# 执行顺序:
# AutogradCPU -> CPU (实际计算)
# 反向传播时:
# 梯度计算在 Autograd 层处理

# 注册 Autograd 实现
@impl("my_op", "AutogradCPU")
def my_op_autograd(ctx, x):
    # 前向
    result = my_op_impl(x)
    
    # 设置反向传播
    ctx.save_for_backward(x, result)
    return result

@impl("my_op", "AutogradCPU", backward=True)
def my_op_backward(ctx, grad_output):
    x, result = ctx.saved_tensors
    grad_input = grad_output * 2  # 示例
    return grad_input
```

### 5.2 CompositeExplicitAutograd

```python
# CompositeExplicitAutograd 用于复合实现
# 算子被分解为其他基本算子的组合

# 示例：gelu 可以用基本算子实现
@impl("gelu", "CompositeExplicitAutograd")
def gelu_composite(x):
    # GELU(x) = x * Φ(x) 其中 Φ 是标准正态 CDF
    # 使用近似: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    import math
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    ))
```

### 5.3 Meta 设备

```python
# Meta 设备用于抽象执行，不分配实际内存
# 用于形状推导、图优化等场景

@impl("my_op", "Meta")
def my_op_meta(x):
    # 只返回形状信息，不实际计算
    return torch.empty_like(x, device='meta')

# 使用 Meta 设备进行形状推导
with torch.device('meta'):
    x = torch.randn(3, 4)
    y = my_op(x)
    print(y.shape)  # torch.Size([3, 4])
    print(y.data_ptr())  # 0 (没有实际内存)
```

### 5.4 Batched / Vmap

```python
# FuncTorchBatched / FuncTorchVmap
# 用于批量处理和向量化映射

from torch.func import vmap

def predict(params, batch):
    # 处理单个样本
    return model(params, batch)

# 自动向量化
batched_predict = vmap(predict, in_dims=(None, 0))
results = batched_predict(params, batches)  # 批量处理
```

### 5.5 Quantized（量化）

```python
# Quantized Dispatch Key 用于量化算子

@impl("add", "Quantized")
def add_quantized(x, y):
    # 量化加法实现
    # 处理 scale 和 zero_point
    x_scale, x_zero = x.q_scale(), x.q_zero_point()
    y_scale, y_zero = y.q_scale(), y.q_zero_point()
    
    # 量化算术
    return torch.ops.quantized.add(x, y)
```

### 5.6 Sparse（稀疏）

```python
# Sparse Dispatch Key 用于稀疏张量

@impl("mm", "Sparse")
def mm_sparse(a, b):
    # 稀疏矩阵乘法
    # 只处理非零元素
    return torch.sparse.mm(a, b)
```

---

## 06. 获取与检查 Kernel

### 6.1 get_kernel

**源码位置**: `torch/library.py:1551-1595`

```python
def get_kernel(
    op: str,
    dispatch_key: str | DispatchKey
) -> Callable:
    """
    获取算子在特定 DispatchKey 上的实现
    
    Args:
        op: 算子名称（如 "aten::add.Tensor"）
        dispatch_key: DispatchKey（如 "CPU", "CUDA"）
    
    Returns:
        算子实现函数
    
    Example::
        
        # 获取 CPU 实现
        cpu_impl = torch.library.get_kernel("aten::add.Tensor", "CPU")
        
        # 获取 CUDA 实现
        cuda_impl = torch.library.get_kernel("aten::add.Tensor", "CUDA")
        
        # 使用 DispatchKey 枚举
        from torch._C import DispatchKey
        cpu_impl = torch.library.get_kernel("aten::add.Tensor", DispatchKey.CPU)
    """
    # 实际实现从 C++ 层获取
    pass
```

### 6.2 has_kernel

```python
def has_kernel(op: str, dispatch_key: str | DispatchKey) -> bool:
    """
    检查算子是否有特定 DispatchKey 的实现
    
    Returns:
        bool
    """
    pass
```

### 6.3 检查注册状态

```python
# 查看算子的所有实现
op = torch.ops.aten.add.default
print(op)  # OpOverload(qualified_name='aten.add.default')

# 查看所有重载
for overload in torch.ops.aten.add.overloads():
    print(overload)

# 查看特定 DispatchKey 的实现
for key in ['CPU', 'CUDA', 'CompositeExplicitAutograd']:
    if torch.library.has_kernel('aten::add.Tensor', key):
        print(f"{key}: registered")
    else:
        print(f"{key}: not registered")
```

---

## 07. 自定义算子实战

### 7.1 定义并实现自定义算子

```python
import torch
from torch.library import Library, impl

# 1. 创建库
my_lib = Library("my_ops", "DEF")

# 2. 定义算子签名
my_lib.define("double(Tensor x) -> Tensor")
my_lib.define("add_scalar(Tensor x, Scalar s) -> Tensor")

# 3. 实现 CPU 版本
@impl(my_lib, "double", "CPU")
def double_cpu(x):
    return x * 2

@impl(my_lib, "add_scalar", "CPU")
def add_scalar_cpu(x, s):
    return x + s

# 4. 实现 CUDA 版本（如果有）
if torch.cuda.is_available():
    @impl(my_lib, "double", "CUDA")
    def double_cuda(x):
        return x * 2  # 实际应使用 CUDA kernel
    
    @impl(my_lib, "add_scalar", "CUDA")
    def add_scalar_cuda(x, s):
        return x + s

# 5. 实现 Autograd 版本
@impl(my_lib, "double", "AutogradCPU")
def double_autograd(ctx, x):
    result = x * 2
    ctx.save_for_backward(x)
    return result

@impl(my_lib, "double", "AutogradCPU", backward=True)
def double_backward(ctx, grad_output):
    x, = ctx.saved_tensors
    return grad_output * 2  # d(2x)/dx = 2

# 6. 使用自定义算子
from torch import my_ops

x = torch.randn(3, requires_grad=True)
y = my_ops.double(x)  # 调用自定义算子
y.sum().backward()
print(x.grad)  # 应该全为 2
```

### 7.2 注册 Python 实现

```python
# 使用 Python 实现（较慢，但易于开发）

from torch.library import impl, PyCustomOpDef

# 定义 Python 自定义算子
lib = Library("my_lib", PyCustomOpDef)

@lib.impl("my_function", "Python")
def my_function_impl(x):
    return torch.sigmoid(x) * x  # SiLU 激活

# Python 实现自动支持 Autograd
```

---

## 8. Dispatch 调试工具

### 8.1 查看 Dispatch 日志

```bash
# 启用 Dispatch 日志
TORCH_LOGS=dispatch python script.py

# 查看详细日志
TORCH_LOGS=graph_breaks,recompiles,dispatch python script.py
```

### 8.2 使用 Dispatcher 钩子

```python
# 注册 Dispatch 钩子
def dispatch_hook(op, types, args, kwargs):
    print(f"Dispatching: {op}")
    return None  # 返回 None 让正常 Dispatch 继续

# 注意：实际 API 可能随版本变化
```

---

## 附录：DispatchKey 完整列表

**源码位置**: `torch/_C/__init__.pyi.in`

```python
# torch._C.DispatchKey 枚举值
DispatchKey.CPU                    # CPU 实现
DispatchKey.CUDA                   # CUDA 实现
DispatchKey.XLA                    # TPU 实现
DispatchKey.MPS                    # Apple Metal 实现
DispatchKey.PrivateUse1            # 自定义后端 1
DispatchKey.PrivateUse2            # 自定义后端 2
DispatchKey.IPU                    # Graphcore IPU
DispatchKey.XPU                    # Intel GPU
DispatchKey.HPU                    # Habana Gaudi
DispatchKey.VE                     # NEC Vector Engine
DispatchKey.MTIA                   # Meta AI 加速器
DispatchKey.Lazy                   # 惰性执行
DispatchKey.Meta                   # 抽象执行（元张量）

# 自动微分
DispatchKey.AutogradCPU
DispatchKey.AutogradCUDA
DispatchKey.AutogradXLA
DispatchKey.AutogradNestedTensor
DispatchKey.AutogradDispatchKey

# 复合实现
DispatchKey.CompositeExplicitAutograd
DispatchKey.CompositeExplicitAutogradNonFunctional
DispatchKey.CompositeExplicitDeviceDispatch

# 特殊功能
DispatchKey.Quantized              # 量化
DispatchKey.SparseCsr              # 稀疏 CSR
DispatchKey.Sparse                 # 稀疏
DispatchKey.Mkldnn                 # MKL-DNN
DispatchKey.GLSL                   # OpenGL

# 自动混合精度
DispatchKey.AutocastCPU
DispatchKey.AutocastCUDA

# 函数式/批量处理
DispatchKey.FuncTorchBatched       # 批量
DispatchKey.FuncTorchVmap          # 向量化
DispatchKey.FuncTorchGrad          # 梯度
DispatchKey.FuncTorchDynamicShapeBackwards

# 调试/分析
DispatchKey.Named                  # 命名维度
DispatchKey.Concrete               # 具体张量
DispatchKey.Symbolic               # 符号执行

# 回退
DispatchKey.BackendSelect          # 后端选择
DispatchKey.Zero                   # 零张量
```

---

## 后续章节

- [07. Tensor 子类实战](./07-tensor-subclass.md) - 自定义 Tensor 实现
- [08. 设备管理](./08-device-management.md) - 设备抽象与管理
