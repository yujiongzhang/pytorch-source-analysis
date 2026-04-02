# 01. 整体架构概览

> 本文档深入解析 PyTorch Python 前端与 Tensor 系统的整体架构
> 
> **阅读建议**: 本文档是系列的起点，建议先阅读再深入后续章节

---

## 01. Python 前端在 PyTorch 中的位置

### 1.1 PyTorch 的三层架构

PyTorch 采用 **Python 前端 + C++ 核心** 的混合架构，理解这一架构对于阅读源码至关重要。

```
┌─────────────────────────────────────────────────────────────┐
│                    用户 Python 代码                          │
│                 print(x + y), loss.backward()               │
├─────────────────────────────────────────────────────────────┤
│  Python 前端层 (torch/目录)                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  torch/_tensor.py    │ Tensor 类定义 (本文档核心)      │  │
│  │  torch/_ops.py       │ 算子注册与重载                   │  │
│  │  torch/overrides.py  │ __torch_function__ 协议         │  │
│  │  torch/storage.py    │ Storage 存储系统                │  │
│  │  torch/autograd/     │ 自动微分 Python 接口            │  │
│  │  torch/library.py    │ 算子注册 API                     │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  pybind11 / C++ 绑定层 (torch._C)                           │
│  - Python 对象与 C++ 对象的桥梁                              │
│  - 负责类型转换和方法绑定                                    │
├─────────────────────────────────────────────────────────────┤
│  C++ 核心层 (torch/csrc/ 目录)                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  ATen              │ 张量算子库 (Add, MatMul 等)       │  │
│  │  torch/csrc/autograd │ 自动微分引擎 (反向传播)         │  │
│  │  c10/              │ 核心抽象 (Device, ScalarType 等)  │  │
│  │  aten/             │ ATen 算子实现                      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**关键点说明**:

1. **Python 层主要负责**:
   - API 设计和用户接口
   - 运算符重载 (`__add__`, `__mul__` 等)
   - 动态特性支持（如 `__torch_function__`）
   - 自动微分的 Python 接口

2. **C++ 层主要负责**:
   - 实际计算执行（算子内核）
   - 内存管理
   - 设备抽象（CPU、CUDA 等）
   - 反向传播引擎

3. **边界在哪里**:
   - 许多 Tensor 方法在 Python 层声明，但实现在 C++ 层
   - 例如：`tensor.matmul()` 在 Python 层调用，但实际计算在 C++ ATen

---

## 02. Tensor 类的继承体系

### 2.1 Tensor 类的定义

**源码位置**: `torch/_tensor.py:110`

```python
# NB: If you add a new method to Tensor, you must update
# torch/_C/__init__.pyi.in to add a type annotation for your method;
# otherwise, it will not show up in autocomplete.
class Tensor(torch._C.TensorBase):
    _is_param: bool
    __dlpack_c_exchange_api__: object = torch._C._dlpack_exchange_api()
    
    def _clear_non_serializable_cached_data(self):
        """清除缓存数据以支持序列化"""
        ...
    
    def __deepcopy__(self, memo):
        """深度拷贝实现"""
        ...
```

**关键观察**:

1. `Tensor` 继承自 `torch._C.TensorBase`
2. `torch._C.TensorBase` 是 C++ 扩展类，通过 pybind11 绑定
3. `Tensor` 类本身几乎是一个"空壳"，大部分方法来自 C++ 继承

### 2.2 完整的继承关系图

```
object (Python 基类)
  │
  └─ torch._C.TensorBase (C++ 扩展类)
       │
       └─ Tensor (torch/_tensor.py:110)  ← 我们主要使用的类
            │
            ├─ Parameter (torch/nn/parameter.py:30)  ← 神经网络参数
            │
            └─ 其他子类 (如自定义 Tensor)
```

### 2.3 继承体系的两个关键概念

#### 概念 1：C++ TensorBase 提供核心方法

`torch._C.TensorBase` (C++ 层) 提供了 Tensor 的核心方法：

```cpp
// C++ 层定义（简化版）
// 位置：torch/csrc/autograd/python_variable.cpp

class TensorBase {
public:
    // 基本属性访问
    int64_t dim() const;           // 维度数
    IntArrayRef sizes() const;     // 形状
    IntArrayRef strides() const;   // 步幅
    
    // 数据类型和设备
    ScalarType scalar_type() const;  // dtype
    Device device() const;           // device
    
    // 内存相关
    void* data_ptr() const;          // 数据指针
    int64_t numel() const;           // 元素数量
    int64_t element_size() const;    // 单元素字节数
    
    // 算子方法
    Tensor add(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    // ... 数百个算子方法
};
```

这些方法通过 pybind11 暴露给 Python，成为 `Tensor` 类的方法。

#### 概念 2：Python Tensor 添加动态特性

Python 层的 `Tensor` 类主要添加：

1. **Python 特殊方法**: `__repr__`, `__str__`, `__deepcopy__` 等
2. **动态协议**: `__torch_function__` 支持
3. **便捷方法**: 封装 C++ 方法的 Python 友好版本

```python
# torch/_tensor.py 中定义的 Python 特有方法
class Tensor(torch._C.TensorBase):
    def __repr__(self):
        """友好的字符串表示"""
        return torch._tensor_str._str(self)
    
    def __deepcopy__(self, memo):
        """深度拷贝 - Python 特有协议"""
        ...
    
    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        """反向传播 - 调用 C++ 引擎"""
        torch.autograd.backward(self, gradient, retain_graph, create_graph)
```

---

## 03. Storage 存储系统层次

### 3.1 Tensor 与 Storage 的关系

理解 Storage 是理解 PyTorch 内存管理的关键。

**核心概念**: Tensor 只是 Storage 的一个"视图"或"窗口"。

```
┌─────────────────────────────────────────┐
│  Tensor (逻辑视图)                      │
│  ┌─────────────────────────────────┐    │
│  │  size: [3, 4]                   │    │
│  │  stride: (4, 1)                 │    │
│  │  storage_offset: 0              │    │
│  │  dtype: float32                 │    │
│  │  requires_grad: False           │    │
│  └─────────────────────────────────┘    │
│              │                          │
│              │ "看向"                     │
│              ▼                          │
├─────────────────────────────────────────┤
│  Storage (实际数据)                     │
│  ┌─────────────────────────────────┐    │
│  │  [1.0, 2.0, 3.0, 4.0,          │    │
│  │   5.0, 6.0, 7.0, 8.0,          │    │
│  │   9.0, 10.0, 11.0, 12.0]       │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**为什么这样设计**:

1. **内存效率**: 多个 Tensor 可以共享同一块 Storage（如切片操作）
2. **灵活性**: 同一块数据可以有不同形状/步幅的视图
3. **设备管理**: Storage 负责实际的设备内存分配

### 3.2 Storage 的层次结构

**源码位置**: `torch/storage.py`

PyTorch 的 Storage 系统经历了从 TypedStorage 到 UntypedStorage 的演进：

```
历史演进:
  PyTorch < 1.8: 只有 TypedStorage (如 FloatStorage, IntStorage)
  PyTorch 1.8+:   引入 UntypedStorage 作为底层存储
  PyTorch 2.x:    TypedStorage 已弃用，但为向后兼容仍保留

当前架构:
  Tensor
    │
    └─> _typed_storage() [已弃用，但仍在使用]
          │
          └─> TypedStorage (带 dtype 信息，包装层)
                │
                └─> UntypedStorage (底层字节数组，实际存储)
```

### 3.3 _StorageBase 基类

**源码位置**: `torch/storage.py:41`

```python
class _StorageBase:
    """所有 Storage 类的抽象基类"""
    
    _cdata: Any  # C++ 存储对象的指针
    is_sparse: bool = False
    is_sparse_csr: bool = False
    device: torch.device
    _fake_device: torch.device | None = None  # FakeTensor 用
    _checkpoint_offset: int | None = None  # 序列化用
    
    def __len__(self) -> int:
        """返回存储的元素数量"""
        raise NotImplementedError
    
    def __getitem__(self, idx):
        """通过索引访问元素"""
        raise NotImplementedError
    
    def __setitem__(self, *args, **kwargs):
        """通过索引设置元素"""
        raise NotImplementedError
    
    def copy_(self, source: T, non_blocking=False) -> T:
        """从另一个存储复制数据"""
        raise NotImplementedError
    
    def nbytes(self) -> int:
        """返回存储占用的字节数"""
        raise NotImplementedError
    
    def size(self) -> int:
        """返回存储大小（字节数）"""
        return self.nbytes()
    
    def element_size(self) -> int:
        """返回每个元素的字节数"""
        raise NotImplementedError
    
    def data_ptr(self) -> int:
        """返回底层数据指针（内存地址）"""
        raise NotImplementedError
    
    def resizable(self) -> bool:
        """返回存储是否可调整大小"""
        raise NotImplementedError
    
    def resize_(self, size: int):
        """原地调整存储大小"""
        raise NotImplementedError
```

**关键属性说明**:

| 属性 | 说明 |
|------|------|
| `_cdata` | 指向 C++ 存储对象的指针，实际数据在这里 |
| `device` | Storage 所在的设备 (cpu, cuda:0 等) |
| `is_sparse` | 是否是稀疏存储 |
| `_fake_device` | FakeTensor 模式下使用 |

### 3.4 TypedStorage 类（已弃用但仍在用）

**源码位置**: `torch/storage.py`

```python
class TypedStorage:
    """
    带类型信息的 Storage（已弃用，但为了向后兼容仍在使用）
    
    .. warning::
        TypedStorage is deprecated. It will be removed in the future, and
        UntypedStorage will be the only storage class.
    """
    
    def __init__(self, wrap_storage, dtype, _internal=False):
        """
        Args:
            wrap_storage: 底层 UntypedStorage
            dtype: 数据类型
            _internal: 内部使用标志（避免弃用警告）
        """
        self._untyped_storage = wrap_storage
        self.dtype = dtype
    
    def __repr__(self):
        info_str = f"[{torch.typename(self)}(device={self.device}) of size {len(self)}]"
        if self.device.type == "meta":
            return "...\n" + info_str
        data_str = " " + "\n ".join(str(self[i]) for i in range(self.size()))
        return data_str + "\n" + info_str
    
    def clone(self):
        """返回此存储的副本"""
        return type(self)(self.nbytes(), device=self.device).copy_(self)
    
    def _share_memory_(self):
        """将存储移到共享内存"""
        with _share_memory_lock:
            # ... 共享内存实现
            pass
```

**为什么弃用 TypedStorage**:

- TypedStorage 为每种 dtype 维护一个类型（FloatStorage, IntStorage 等），导致代码重复
- UntypedStorage 统一管理，dtype 信息由 Tensor 层维护更清晰

### 3.5 UntypedStorage 类

**源码位置**: `torch/storage.py`

```python
class UntypedStorage:
    """
    无类型底层存储 - 原始字节数组
    
    这是未来唯一的 Storage 类型
    """
    
    def __new__(cls, size_or_sequence=None, device=None, dtype=None):
        """
        创建 UntypedStorage
        
        Args:
            size_or_sequence: 大小或序列
            device: 设备类型
            dtype: (可选) 用于推断元素大小的数据类型
        """
        # 实际实现在 C++ 层
        pass
    
    @classmethod
    def from_file(cls, filename, shared=False, nbytes=None):
        """从文件创建存储"""
        pass
    
    @classmethod
    def from_buffer(cls, buffer):
        """从缓冲区（如 numpy 数组）创建存储"""
        pass
```

---

## 04. 关键模块文件一览

### 4.1 核心文件索引

下表列出 Python 前端的核心文件及其作用：

| 文件 | 核心内容 | 关键行号/类 |
|------|----------|-------------|
| `torch/_tensor.py` | Tensor 类定义 | L110: `class Tensor` |
| `torch/storage.py` | Storage 系统 | L41: `_StorageBase` |
| `torch/overrides.py` | `__torch_function__` 协议 | `handle_torch_function` |
| `torch/nn/parameter.py` | Parameter 类 | L30: `class Parameter` |
| `torch/_ops.py` | 算子注册系统 | `OpOverloadPacket` |
| `torch/library.py` | 算子注册 API | `Library` 类 |
| `torch/_refs/__init__.py` | 工厂函数引用实现 | `zeros`, `ones` 等 |
| `torch/_C/__init__.pyi.in` | C++ 绑定类型注解 | `TensorBase` 类型定义 |

### 4.2 Autograd 相关文件

| 文件 | 核心内容 |
|------|----------|
| `torch/autograd/__init__.py` | `backward()`, `grad()` 等入口 |
| `torch/autograd/function.py` | `Function` 基类 |
| `torch/autograd/grad_mode.py` | `no_grad`, `enable_grad` 等 |
| `torch/autograd/variable.py` | `Variable` 别名（历史遗留） |

### 4.3 C++ 核心文件（供参考）

| 文件 | 核心内容 |
|------|----------|
| `torch/csrc/autograd/python_variable.cpp` | C++ Variable 定义 |
| `torch/csrc/autograd/engine.cpp` | 反向传播引擎 |
| `torch/csrc/autograd/variable.h` | Variable C++ 头文件 |
| `torch/csrc/autograd/function.h` | Function C++ 定义 |

---

## 05. Python 与 C++ 的边界

### 5.1 方法实现位置的三种模式

PyTorch 中方法的实现遵循三种模式：

#### 模式 1：Python 纯实现

完全在 Python 层实现的方法，通常涉及 Python 特有协议：

```python
# torch/_tensor.py
def __deepcopy__(self, memo):
    """深度拷贝 - 纯 Python 实现"""
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.__deepcopy__, (self,), self, memo)
    if not self.is_leaf:
        raise RuntimeError("Only Tensors created explicitly by the user...")
    # ... Python 实现逻辑
    return new_tensor
```

#### 模式 2：Python 包装，C++ 实现

Python 层作为包装，实际计算在 C++ 层：

```python
# torch/_tensor.py
def backward(self, gradient=None, retain_graph=None, create_graph=False):
    """反向传播 - Python 包装"""
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.backward, (self,), self, ...)
    # 调用 C++ 引擎
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
```

#### 模式 3：直接绑定 C++ 方法

直接引用 C++ 方法，无 Python 包装：

```python
# torch/_tensor.py
__neg__ = _C.TensorBase.neg  # 直接绑定
__abs__ = _C.TensorBase.abs  # 直接绑定
detach = _C.TensorBase.detach  # 直接绑定
```

### 5.2 如何判断方法的实现位置

```python
import torch

x = torch.randn(3, 4)

# 查看方法来源
print(x.__deepcopy__.__module__)  # 'torch._tensor' - Python 实现
print(x.neg.__module__)           # 可能为空或 'torch._C' - C++ 绑定

# 查看方法类型
print(type(x.__deepcopy__))       # <class 'method'>
print(type(x.neg))                # <class 'builtin_function_or_method'>
```

### 5.3 类型注解的位置

**重要提示**: 如果在 Tensor 类中添加新方法，必须更新类型注解文件：

```python
# torch/_C/__init__.pyi.in:1953
class TensorBase(metaclass=_TensorMeta):
    """C++ TensorBase 的 Python 类型注解"""
    
    def matmul(self, other: Tensor) -> Tensor: ...
    def add(self, other: Tensor) -> Tensor: ...
    # ... 所有 C++ 方法的类型注解
```

**为什么需要类型注解**:
- C++ 方法的签名无法自动推断
- 类型注解提供 IDE 自动补全
- 类型检查工具（如 mypy）依赖这些注解

---

## 06. Parameter 类详解

### 6.1 Parameter 的设计目的

`Parameter` 是 `Tensor` 的子类，用于神经网络的参数。

**设计动机**: 
- 区分"模型参数"和"临时状态"
- 参数自动加入 `Module.parameters()` 迭代器
- 临时状态（如 RNN 隐藏状态）不作为参数

### 6.2 Parameter 的实现

**源码位置**: `torch/nn/parameter.py:30`

```python
# Metaclass to combine _TensorMeta and the instance check override for Parameter.
class _ParameterMeta(torch._C._TensorMeta):
    # Make `isinstance(t, Parameter)` return True for custom tensor instances 
    # that have the _is_param flag.
    def __instancecheck__(self, instance) -> bool:
        if self is Parameter:
            if isinstance(instance, torch.Tensor) and getattr(
                instance, "_is_param", False
            ):
                return True
        return super().__instancecheck__(instance)


class Parameter(torch.Tensor, metaclass=_ParameterMeta):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        
        # 路径 1: 标准 Tensor 或 Parameter - 使用 _make_subclass
        if type(data) is torch.Tensor or type(data) is Parameter:
            return torch.Tensor._make_subclass(cls, data, requires_grad)
        
        # 路径 2: 自定义 Tensor 类型 - 设置 _is_param 标志
        t = data.detach().requires_grad_(requires_grad)
        if type(t) is not type(data):
            raise RuntimeError(
                f"Creating a Parameter from an instance of type {type(data).__name__} "
                "requires that detach() returns an instance of the same type..."
            )
        t._is_param = True
        return t
    
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        result = type(self)(
            self.data.clone(memory_format=torch.preserve_format), 
            self.requires_grad
        )
        memo[id(self)] = result
        return result
    
    def __repr__(self) -> str:
        return "Parameter containing:\n" + super().__repr__()
    
    # 禁用 __torch_function__ 以减少开销
    __torch_function__ = _disabled_torch_function_impl
```

### 6.3 Parameter 的关键特性

1. **`_is_param` 标志**: 自定义 Tensor 类型通过此标志标记为参数
2. **`isinstance` 重载**: 元类使 `isinstance(t, Parameter)` 对 `_is_param=True` 的 Tensor 返回 `True`
3. **禁用 `__torch_function__`**: 减少性能开销

### 6.4 Parameter 使用示例

```python
import torch
import torch.nn as nn

# 创建 Parameter
weight = nn.Parameter(torch.randn(3, 4))
bias = nn.Parameter(torch.zeros(3))

# 在 Module 中使用
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4))  # 自动注册为参数
        self.buffer = torch.zeros(3)  # 不是参数，只是缓存
    
    def forward(self, x):
        return x @ self.weight.t()

model = MyModel()

# 查看参数
for name, param in model.named_parameters():
    print(name, type(param))  # weight <class 'torch.nn.parameter.Parameter'>

# buffer 不会出现在参数列表中
```

---

## 07. __torch_function__ 协议简介

### 7.1 什么是 __torch_function__

`__torch_function__` 是 PyTorch 的协议，允许自定义 Tensor 子类拦截 torch 函数调用。

**类比**: 类似于 NumPy 的 `__array_function__` 协议

### 7.2 基本用法

```python
import torch

class LoggingTensor(torch.Tensor):
    """一个记录所有操作的 Tensor 子类"""
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        # 记录函数调用
        print(f"Calling {func.__name__}")
        
        # 调用原始实现
        result = func(*args, **kwargs)
        
        # 将结果包装回 LoggingTensor
        if isinstance(result, torch.Tensor):
            result = LoggingTensor(result)
        return result

# 使用示例
x = LoggingTensor([1.0, 2.0, 3.0])
y = x + 1  # 打印：Calling add.Tensor
z = torch.sum(y)  # 打印：Calling sum
```

### 7.3 handle_torch_function 函数

**源码位置**: `torch/overrides.py`

```python
def handle_torch_function(
    public_api,
    relevant_types,
    args,
    kwargs,
):
    """
    检查参数是否有 __torch_function__ 实现并调用它们
    
    参数:
        public_api: 被调用的公共 API 函数
        relevant_types: 所有参数的类型
        args: 位置参数
        kwargs: 关键字参数
    """
    # 1. 检查 __torch_function__ 模式
    if _is_torch_function_mode_enabled():
        mode = _get_torch_function_mode()
        result = mode.__torch_function__(public_api, types, args, kwargs)
        if result is not None:
            return result
    
    # 2. 检查参数是否有 __torch_function__ 方法
    overloaded_args = _find_torch_function_overloads(args, kwargs)
    if overloaded_args:
        torch_func_method = overloaded_args[0].__torch_function__
        result = torch_func_method(public_api, types, args, kwargs)
        if result is not None:
            return result
    
    # 3. 没有覆盖，调用原始实现
    return public_api(*args, **kwargs)
```

### 7.4 为什么 Parameter 禁用 __torch_function__

```python
class Parameter(torch.Tensor):
    __torch_function__ = _disabled_torch_function_impl

# _disabled_torch_function_impl 返回 NotImplemented，让原始函数继续执行
def _disabled_torch_function_impl(cls, func, types, args=(), kwargs=None):
    return NotImplemented
```

**原因**:
- Parameter 在训练循环中被频繁访问
- `__torch_function__` 检查带来性能开销
- Parameter 不需要自定义行为，禁用可提升性能

---

## 08. 源码阅读技巧

### 8.1 使用 rg/grep 搜索源码

```bash
# 定位类定义
rg "class Tensor" torch/_tensor.py
rg "class _StorageBase" torch/storage.py

# 定位方法定义
rg "def backward" torch/_tensor.py
rg "def __torch_dispatch__" torch/

# 搜索特定模式
rg "has_torch_function" torch/_tensor.py
rg "@_handle_torch_function" torch/_tensor.py

# 统计文件行数
wc -l torch/_tensor.py  # 约 1400 行
```

### 8.2 理解注释中的约定

PyTorch 源码中的注释遵循特定约定：

```python
# NB: 开头注释表示"重要说明"(Note Bien)
# NB: If you add a new method to Tensor, you must update
# torch/_C/__init__.pyi.in to add a type annotation

# TODO: 开头表示待办事项
# TODO: Implement support for sparse tensors

# See: 引用相关问题或 PR
# See https://github.com/pytorch/pytorch/issues/12345
```

### 8.3 查看方法解析顺序

```python
import torch

# 查看 Tensor 的方法解析顺序 (MRO)
print(torch.Tensor.__mro__)
# (<class 'torch.Tensor'>, <class 'torch._C.TensorBase'>, <class 'object'>)

# 查看方法来源
import inspect
print(inspect.getfile(torch.Tensor.__deepcopy__))  # Python 文件
print(inspect.getfile(torch.Tensor.neg))           # C++ 扩展
```

---

## 附录：关键术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| 张量 | Tensor | 多维数组，PyTorch 的核心数据结构 |
| 存储 | Storage | Tensor 底层数据的连续内存块 |
| 变量 | Variable | 历史概念，现已与 Tensor 合并 |
| 参数 | Parameter | 用于神经网络参数的 Tensor 子类 |
| 算子 | Operator/Op | 对 Tensor 执行的操作（如 add, matmul） |
| 重载 | Overload | 同一算子针对不同设备的不同实现 |
| 调度 | Dispatch | 将算子调用路由到正确实现的过程 |
| 自动微分 | Autograd | 自动计算梯度的系统 |
| 计算图 | Computation Graph | 记录计算历史的数据结构 |
| 后端 | Backend | 针对特定设备的实现（如 CPU、CUDA） |

---

## 后续章节

- [02. Tensor 操作与方法](./02-tensor-operations.md) - 运算符重载、索引切片、类型转换
- [03. Autograd 自动微分](./03-autograd.md) - 计算图与反向传播引擎
- [04. Storage 与内存管理](./04-storage-memory.md) - 底层存储与内存分配
- [05. 工厂函数](./05-factory-functions.md) - Tensor 创建机制
- [06. Dispatcher 调度系统](./06-dispatcher.md) - Dispatch Key 机制

---

## 学习检查

阅读完本文档后，你应该能够回答：

1. Tensor 类继承自哪个 C++ 类？
2. Tensor 和 Storage 是什么关系？
3. TypedStorage 和 UntypedStorage 有什么区别？
4. Parameter 类的设计目的是什么？
5. `__torch_function__` 协议的作用是什么？
6. 如何判断一个方法是 Python 实现还是 C++ 绑定？

如果这些问题都能回答，说明你已经掌握了 PyTorch Python 前端的基本架构！
