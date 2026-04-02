# 05. 工厂函数与 Tensor 创建

> 本文档解析 PyTorch 工厂函数的实现机制与 Tensor 创建流程

---

## 01. 工厂函数概述

### 1.1 什么是工厂函数

工厂函数是创建 Tensor 的函数，如 `torch.zeros()`, `torch.randn()`, `torch.ones()` 等。

```python
# 常见的工厂函数
torch.zeros(3, 4)           # 全 0 Tensor
torch.ones(3, 4)            # 全 1 Tensor
torch.empty(3, 4)           # 未初始化 Tensor
torch.randn(3, 4)           # 标准正态分布
torch.arange(0, 10)         # 等差数列
torch.linspace(0, 1, 5)     # 等间距数列
torch.eye(3)                # 单位矩阵
torch.full((3, 4), 5.0)     # 填充指定值
```

### 1.2 工厂函数的实现层次

```
用户调用 torch.zeros(3, 4)
       ↓
torch/_torch_docs.py (文档字符串)
       ↓
torch/_refs/__init__.py (引用实现)
       ↓
torch/_ops.py (算子注册)
       ↓
torch.ops.aten.zeros.default (ATen 算子)
       ↓
C++ 实现 (ATen)
```

---

## 02. 工厂函数的实现位置

### 2.1 torch/_refs/__init__.py

**源码位置**: `torch/_refs/__init__.py`

这是工厂函数的"引用实现"（reference implementation），使用纯 Python 编写，作为标准参考。

```python
# torch/_refs/__init__.py

def zeros(*size: int, dtype=None, device=None, requires_grad=False):
    """
    创建全 0 Tensor
    
    Args:
        size: Tensor 形状
        dtype: 数据类型
        device: 设备类型
        requires_grad: 是否需要梯度
    """
    # 调用 ATen 算子
    return torch.ops.aten.zeros.default(size, dtype, device, requires_grad)


def ones(*size: int, dtype=None, device=None, requires_grad=False):
    """创建全 1 Tensor"""
    return torch.ops.aten.ones.default(size, dtype, device, requires_grad)


def empty(*size: int, dtype=None, device=None, requires_grad=False):
    """创建未初始化 Tensor"""
    return torch.ops.aten.empty.default(size, dtype, device, requires_grad)


def full(
    size: int,
    fill_value: Number,
    dtype=None,
    device=None,
    requires_grad=False,
):
    """创建填充指定值的 Tensor"""
    return torch.ops.aten.full.default(
        size, fill_value, dtype, device, requires_grad
    )
```

### 2.2 常见工厂函数列表

**源码位置**: `torch/_refs/__init__.py:59-150`

```python
__all__ = [
    # 基础工厂函数
    "zeros",
    "ones", 
    "empty",
    "full",
    "zeros_like",
    "ones_like",
    "empty_like",
    "full_like",
    
    # 随机数生成
    "rand",
    "randn",
    "randint",
    "randperm",
    
    # 序列生成
    "arange",
    "linspace",
    "logspace",
    "geometric",
    
    # 特殊矩阵
    "eye",
    "diag",
    "diag_embed",
    "triangular",
    
    # 索引相关
    "index_add",
    "index_copy",
    "index_select",
    "index_fill",
    
    # ... 更多函数
]
```

### 2.3 文档字符串定义

**源码位置**: `torch/_torch_docs.py`

```python
# torch/_torch_docs.py

def parse_kwargs(desc):
    """解析参数文档"""
    regx = re.compile(r"\n\s{4}(?!\s)")
    kwargs = [section.strip() for section in regx.split(desc)]
    return {desc.split(" ")[0]: desc for desc in kwargs}


# 公共参数定义
common_args = parse_kwargs("""
    input (Tensor): the input tensor.
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator
    out (Tensor, optional): the output tensor.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format
""")

# 合并参数文档
reduceops_common_args = merge_dicts(
    common_args,
    parse_kwargs("""
    dtype (:class:`torch.dtype`, optional): the desired data type
    keepdim (bool): whether the output tensor has dim retained
""")
)

# 添加文档字符串到 C++ 函数
add_docstr(
    torch._C._VariableFunctions.randn,
    """
    randn(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
    
    Returns a tensor filled with random numbers from a normal distribution.
    """,
)
```

---

## 03. _ops.py 与算子注册

### 3.1 算子注册机制

**源码位置**: `torch/_ops.py`

```python
class OpOverloadPacket:
    """
    算子重载包
    
    一个算子可能有多个重载版本（如 CPU、CUDA、Sparse 等）
    """
    
    def __init__(self, qualified_name):
        self.qualified_name = qualified_name
        self._overloads = {}
    
    def __getattr__(self, overload_name):
        """获取特定重载版本"""
        return self._overloads.get(overload_name)
    
    def __call__(self, *args, **kwargs):
        """默认调用 default 重载"""
        return self.default(*args, **kwargs)


class OpOverload:
    """
    算子的单个重载版本
    
    每个 OpOverload 对应一个具体的实现
    """
    
    def __init__(self, op, overload_name):
        self.op = op
        self.overload_name = overload_name
        self._schema = None
    
    @property
    def _schema(self):
        """获取算子签名"""
        if self._schema_ is None:
            self._schema_ = self.op.schema
        return self._schema_
    
    def __call__(self, *args, **kwargs):
        """调用算子"""
        return op(*args, **kwargs)
```

### 3.2 工厂函数的算子路径

```python
# torch.zeros 的调用链
import torch

# 1. 用户调用
x = torch.zeros(3, 4)

# 2. torch.zeros 定义（实际在 torch/_C 中）
# torch/_C._VariableFunctions.zeros

# 3. 调用 ATen 算子
x = torch.ops.aten.zeros.default([3, 4])

# 查看算子
print(torch.ops.aten.zeros)
# OpOverloadPacket(qualified_name='aten.zeros')

print(torch.ops.aten.zeros.default)
# OpOverloadPacket(qualified_name='aten.zeros.default')
```

---

## 04. 工厂函数分类详解

### 4.1 值填充类

```python
# zeros - 全 0
def zeros(*size, dtype=None, device=None, requires_grad=False):
    return torch.ops.aten.zeros.default(size, dtype, device, requires_grad)

# ones - 全 1
def ones(*size, dtype=None, device=None, requires_grad=False):
    return torch.ops.aten.ones.default(size, dtype, device, requires_grad)

# full - 填充指定值
def full(size, fill_value, dtype=None, device=None, requires_grad=False):
    return torch.ops.aten.full.default(size, fill_value, dtype, device, requires_grad)

# empty - 未初始化（内存中随机值）
def empty(*size, dtype=None, device=None, requires_grad=False):
    return torch.ops.aten.empty.default(size, dtype, device, requires_grad)
```

### 4.2 随机数生成类

**源码位置**: `torch/_refs/__init__.py`

```python
def rand(
    *size,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    """
    均匀分布随机数 [0, 1)
    
    Args:
        size: Tensor 形状
        dtype: 数据类型
        generator: 随机数生成器
    """
    return torch.ops.aten.rand.default(size, dtype, layout, device, requires_grad)


def randn(
    *size,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    """
    标准正态分布随机数 (mean=0, std=1)
    """
    return torch.ops.aten.randn.default(size, dtype, layout, device, requires_grad)


def randint(
    low: int,
    high: int,
    size: int,
    dtype=None,
    device=None,
    requires_grad=False,
):
    """
    指定范围内的随机整数 [low, high)
    """
    return torch.ops.aten.randint.default(low, high, size, dtype, device, requires_grad)


def randperm(n, dtype=None, layout=None, device=None, requires_grad=False):
    """
    0 到 n-1 的随机排列
    """
    return torch.ops.aten.randperm.default(n, dtype, layout, device, requires_grad)
```

### 4.3 序列生成类

```python
def arange(
    start: Number,
    end: Number = None,
    step: Number = 1,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    """
    创建等差数列
    
    Args:
        start: 起始值（如果 end 为 None，则为 0 到 start）
        end: 结束值
        step: 步长
    
    Example::
        >>> torch.arange(5)
        tensor([0, 1, 2, 3, 4])
        >>> torch.arange(1, 10, 2)
        tensor([1, 3, 5, 7, 9])
    """
    if end is None:
        end = start
        start = 0
    return torch.ops.aten.arange.default(start, end, step, dtype, layout, device, requires_grad)


def linspace(
    start: Number,
    end: Number,
    steps: int,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    """
    创建等间距数列
    
    Args:
        start: 起始值
        end: 结束值
        steps: 元素数量
    
    Example::
        >>> torch.linspace(0, 1, 5)
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    """
    return torch.ops.aten.linspace.default(start, end, steps, dtype, layout, device, requires_grad)


def logspace(
    start: Number,
    end: Number,
    steps: int,
    base: float = 10.0,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    """
    创建等比数列（对数间隔）
    
    Example::
        >>> torch.logspace(0, 2, 5)
        tensor([  1.,   3.162,  10.,  31.623, 100.])
    """
    return torch.ops.aten.logspace.default(start, end, steps, base, dtype, layout, device, requires_grad)
```

### 4.4 特殊矩阵类

```python
def eye(n: int, m: int = None, dtype=None, layout=None, device=None, requires_grad=False):
    """
    创建单位矩阵
    
    Args:
        n: 行数
        m: 列数（默认为 n）
    
    Example::
        >>> torch.eye(3)
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])
    """
    if m is None:
        m = n
    return torch.ops.aten.eye.default(n, m, dtype, layout, device, requires_grad)


def diag(diagonal: Tensor, diagonal_offset: int = 0):
    """
    创建对角矩阵或提取对角线
    
    Args:
        diagonal: 对角线元素或输入 Tensor
        diagonal_offset: 对角线偏移量
    
    Example::
        >>> torch.diag(torch.tensor([1, 2, 3]))
        tensor([[1, 0, 0],
                [0, 2, 0],
                [0, 0, 3]])
    """
    return torch.ops.aten.diag.default(diagonal, diagonal_offset)


def triu(input: Tensor, diagonal: int = 0):
    """
    上三角矩阵
    
    Args:
        input: 输入 Tensor
        diagonal: 对角线偏移量
    """
    return torch.ops.aten.triu.default(input, diagonal)


def tril(input: Tensor, diagonal: int = 0):
    """
    下三角矩阵
    """
    return torch.ops.aten.tril.default(input, diagonal)
```

### 4.5 *_like 系列函数

```python
def zeros_like(
    input: Tensor,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None,
):
    """
    创建与输入 Tensor 形状相同的全 0 Tensor
    
    Args:
        input: 参考 Tensor
        dtype: 覆盖的数据类型
        device: 覆盖的设备
        memory_format: 内存格式
    """
    return torch.ops.aten.zeros_like.default(input, dtype, layout, device, requires_grad, memory_format)


def ones_like(input: Tensor, dtype=None, layout=None, device=None, requires_grad=False):
    """创建与输入 Tensor 形状相同的全 1 Tensor"""
    return torch.ops.aten.ones_like.default(input, dtype, layout, device, requires_grad)


def empty_like(input: Tensor, dtype=None, layout=None, device=None, requires_grad=False):
    """创建与输入 Tensor 形状相同的未初始化 Tensor"""
    return torch.ops.aten.empty_like.default(input, dtype, layout, device, requires_grad)


def full_like(input: Tensor, fill_value, dtype=None, layout=None, device=None, requires_grad=False):
    """创建与输入 Tensor 形状相同并填充指定值的 Tensor"""
    return torch.ops.aten.full_like.default(input, fill_value, dtype, layout, device, requires_grad)
```

---

## 05. Tensor 构造函数

### 5.1 torch.tensor() vs torch.Tensor()

```python
# torch.tensor() - 推荐方式
# 从数据创建新 Tensor，推断 dtype
x = torch.tensor([1, 2, 3])           # dtype=torch.int64
x = torch.tensor([1.0, 2.0, 3.0])     # dtype=torch.float32
x = torch.tensor([1, 2], dtype=float) # 指定 dtype

# torch.Tensor() - 旧式构造函数
# 总是创建 float32，参数是形状
y = torch.Tensor(3, 4)  # 3x4 的未初始化 float32 Tensor
y = torch.Tensor([1, 2, 3])  # 从列表创建（不推荐）
```

### 5.2 torch.Tensor._make_subclass

**源码位置**: `torch/_tensor.py`

```python
class Tensor:
    @staticmethod
    def _make_subclass(cls, data, requires_grad=False):
        """
        创建 Tensor 子类的内部方法
        
        Args:
            cls: 目标子类
            data: 原始 Tensor 数据
            requires_grad: 是否需要梯度
        
        Returns:
            cls 类型的 Tensor
        """
        # 实际实现在 C++ 层
        pass
```

---

## 06. 工厂函数的设备与类型推断

### 6.1 默认设备与类型

```python
# 获取默认配置
torch.get_default_dtype()      # torch.float32
torch.get_default_device()     # device(type='cpu')

# 设置默认配置
torch.set_default_dtype(torch.float64)
torch.set_default_device('cuda')  # 需要 CUDA 可用
```

### 6.2 类型推断规则

```python
# 工厂函数的类型推断优先级：
# 1. 显式指定的 dtype
# 2. 输入的 dtype（对于 *_like 函数）
# 3. torch.get_default_dtype()

# 示例
torch.zeros(3, 4)                          # float32 (默认)
torch.zeros(3, 4, dtype=torch.int32)       # int32 (指定)
torch.zeros(3, 4, dtype=torch.complex64)   # complex64

# 整数工厂函数有特殊规则
torch.randint(0, 10, (3, 4))               # int64 (随机整数默认)
torch.arange(5)                            # int64 (整数序列)
```

### 6.3 设备推断

```python
# 设备推断优先级：
# 1. 显式指定的 device
# 2. torch.get_default_device()
# 3. 'cpu'

# 示例
torch.zeros(3, 4)                          # CPU
torch.zeros(3, 4, device='cuda:0')         # CUDA:0
torch.zeros(3, 4, device=torch.device('mps'))  # MPS

# from 参数（某些函数）
x = torch.randn(3, 4, device='cuda:0')
torch.zeros_like(x)                        # 继承 x 的设备
torch.zeros(3, 4, device=x.device)         # 显式继承
```

---

## 07. 内存格式控制

### 7.1 memory_format 参数

```python
# 内存格式选项
torch.contiguous_format      # 连续内存（默认）
torch.preserve_format        # 保持输入格式
torch.channels_last          # NHWC 格式（用于卷积）
torch.channels_last_3d       # NDHWC 格式（用于 3D 卷积）

# 使用示例
x = torch.randn(3, 4, memory_format=torch.contiguous_format)

# 保持格式
y = torch.ones_like(x, memory_format=torch.preserve_format)

# 通道优先转通道最后
z = torch.randn(3, 224, 224, memory_format=torch.channels_last)
```

### 7.2 工厂函数的 memory_format

```python
def empty(
    *size,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,  # 默认连续
):
    return torch.ops.aten.empty.memory_format(
        size, dtype, layout, device, requires_grad, pin_memory, memory_format
    )
```

---

## 附录：完整工厂函数清单

### 基础工厂函数

| 函数 | 说明 |
|------|------|
| `zeros()` | 全 0 |
| `ones()` | 全 1 |
| `empty()` | 未初始化 |
| `full()` | 填充指定值 |
| `zeros_like()` | 同形状全 0 |
| `ones_like()` | 同形状全 1 |
| `empty_like()` | 同形状未初始化 |
| `full_like()` | 同形状填充 |

### 随机数生成

| 函数 | 说明 |
|------|------|
| `rand()` | 均匀分布 [0,1) |
| `randn()` | 标准正态分布 |
| `randint()` | 随机整数 |
| `randperm()` | 随机排列 |
| `bernoulli()` | 伯努利分布 |
| `multinomial()` | 多项式分布 |
| `normal()` | 正态分布 |
| `uniform()` | 均匀分布 |

### 序列生成

| 函数 | 说明 |
|------|------|
| `arange()` | 等差数列 |
| `linspace()` | 等间距数列 |
| `logspace()` | 对数间隔数列 |
| `geometric()` | 几何分布 |

### 特殊矩阵

| 函数 | 说明 |
|------|------|
| `eye()` | 单位矩阵 |
| `diag()` | 对角矩阵 |
| `diag_embed()` | 对角嵌入 |
| `triangular()` | 三角矩阵 |
| `triu()` | 上三角 |
| `tril()` | 下三角 |
| `bmm()` | 批量矩阵乘法 |

---

## 后续章节

- [06. Dispatcher 调度系统](./06-dispatcher.md) - Dispatch Key 机制
