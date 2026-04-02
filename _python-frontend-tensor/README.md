# Python 前端与 Tensor 系统 - 文档索引

> 本系列文档深入解析 PyTorch Python 前端与 Tensor 系统的实现原理
> 
> **基于版本**: PyTorch v2.11.0+
> **最后更新**: 2026-04-02

---

## 文档清单

| 序号 | 文档 | 内容概述 | 核心源码 |
|------|------|----------|----------|
| 00 | [文档规划](./00-plan.md) | 整体结构与学习路线 | - |
| 01 | [整体架构](./01-architecture.md) | Tensor 继承体系、核心模块关系、Storage 层次 | `torch/_tensor.py:110` |
| 02 | [Tensor 操作](./02-tensor-operations.md) | 运算符重载、索引切片、类型转换、字符串表示 | `torch/_tensor.py:1114+` |
| 03 | [Autograd](./03-autograd.md) | 自动微分原理、计算图构建、反向传播引擎 | `torch/autograd/` |
| 04 | [Storage 与内存](./04-storage-memory.md) | 底层存储系统、内存分配、跨进程共享 | `torch/storage.py` |
| 05 | [工厂函数](./05-factory-functions.md) | Tensor 创建函数、算子注册机制 | `torch/_refs/__init__.py` |
| 06 | [Dispatcher](./06-dispatcher.md) | Dispatch Key 机制、算子注册系统 | `torch/library.py` |

---

## 快速导航

### 按主题查找

#### Tensor 基础
- [整体架构](./01-architecture.md) - Tensor 是什么，继承关系
- [Tensor 操作](./02-tensor-operations.md) - 如何使用 Tensor 进行计算

#### 底层原理
- [Storage 与内存](./04-storage-memory.md) - Tensor 数据如何存储在内存中
- [Dispatcher](./06-dispatcher.md) - PyTorch 如何调度算子到不同设备

#### 高级功能
- [Autograd](./03-autograd.md) - 自动微分如何工作
- [工厂函数](./05-factory-functions.md) - 如何创建和注册自定义算子

---

## 各文档核心内容详解

### 01. 整体架构 (`01-architecture.md`)

**本文档解答的问题**:
- Tensor 类是如何定义的？它与 C++ 层如何绑定？
- PyTorch 的 Python 前端与 C++ 后端如何协作？
- Storage 和 Tensor 是什么关系？

**核心内容**:
- **Python 前端位置**: 位于用户代码与 C++ 核心之间，负责 API 暴露和操作重载
- **Tensor 继承体系**: `Tensor` → `torch._C.TensorBase` (C++ 扩展类)
- **Storage 层次**: `UntypedStorage` (底层字节数组) ← `TypedStorage` (带类型信息，已弃用)
- **关键协议**: `__torch_function__` 允许自定义 Tensor 子类拦截 torch 函数调用

**关键源码位置**:
```
torch/_tensor.py:110          # Tensor 类定义
torch/storage.py:41           # _StorageBase 基类
torch/overrides.py            # __torch_function__ 协议实现
torch/nn/parameter.py:30      # Parameter 类 (神经网络参数)
```

**适合读者**: 初次接触源码的读者，建议从本文档开始

---

### 02. Tensor 操作与方法 (`02-tensor-operations.md`)

**本文档解答的问题**:
- `x + y` 这样的运算符在底层是如何实现的？
- 为什么 Tensor 支持索引切片？
- 如何将 Tensor 转换为 Python 标量？

**核心内容**:
- **运算符重载**: `__add__`, `__mul__`, `__matmul__` 等算术运算符的实现
- **索引切片**: `__getitem__`, `__setitem__` 的实现机制
- **类型转换**: `__float__`, `__int__`, `__bool__`, `item()` 等方法
- **字符串表示**: `__repr__`, `__str__`, `__format__` 如何生成友好的输出
- **Autograd 方法**: `backward()`, `detach()`, `register_hook()` 的原理

**关键源码位置**:
```
torch/_tensor.py:1114-1186    # 运算符重载
torch/_tensor.py:1187+        # 索引与长度方法
torch/_tensor_str.py          # 字符串格式化
```

**代码示例**:
```python
# 运算符重载装饰器模式
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rsub__(self, other):
    return _C._VariableFunctions.rsub(self, other)
```

**适合读者**: 想了解 Tensor 基本操作实现细节的读者

---

### 03. Autograd 自动微分 (`03-autograd.md`)

**本文档解答的问题**:
- PyTorch 如何自动计算梯度？
- `loss.backward()` 底层发生了什么？
- 如何自定义 autograd 函数？

**核心内容**:
- **梯度管理模式**: `no_grad`, `enable_grad`, `inference_mode` 的实现
- **Function 基类**: 自定义 autograd 的接口
- **反向传播引擎**: C++ 层的 `Engine` 类如何执行反向传播
- **计算图**: `grad_fn` 如何记录计算历史
- **梯度检查**: `gradcheck`, `gradgradcheck` 工具的使用

**关键源码位置**:
```
torch/autograd/__init__.py    # backward 等入口函数
torch/autograd/function.py    # Function 基类
torch/autograd/grad_mode.py   # 梯度管理模式
torch/csrc/autograd/engine.cpp # C++ 反向传播引擎
```

**代码示例**:
```python
# 自定义 Function
class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result
```

**适合读者**: 想深入理解自动微分机制的读者

---

### 04. Storage 与内存管理 (`04-storage-memory.md`)

**本文档解答的问题**:
- Tensor 的数据在内存中如何存储？
- 为什么切片操作不复制数据？
- 如何实现跨进程共享 Tensor？

**核心内容**:
- **Storage 层次**: `_StorageBase` → `TypedStorage` → `UntypedStorage`
- **Tensor 与 Storage 关系**: Tensor 是 Storage 的"视图"，包含 shape/stride 等元数据
- **共享内存**: `share_memory_()` 实现多进程共享
- **CUDA 内存管理**: `empty_cache()`, `memory_allocated()` 等工具
- **Pinned Memory**: 锁定内存用于异步数据传输

**关键源码位置**:
```
torch/storage.py:41           # _StorageBase 基类
torch/storage.py:200+         # TypedStorage / UntypedStorage
torch/cuda/memory.py          # CUDA 内存管理
```

**代码示例**:
```python
# 共享 Storage 的视图
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = x[1:3]  # 不复制数据，共享存储
print(y.storage().data_ptr() == x.storage().data_ptr())  # True
```

**适合读者**: 想了解内存管理和性能优化的读者

---

### 05. 工厂函数 (`05-factory-functions.md`)

**本文档解答的问题**:
- `torch.zeros()`, `torch.randn()` 等函数如何实现？
- 如何注册自定义算子？
- 工厂函数的调用链路是什么？

**核心内容**:
- **实现层次**: 用户调用 → `_refs` 引用实现 → `torch.ops` → C++ ATen
- **算子注册**: `torch.library.Library` 注册机制
- **工厂函数分类**: 值填充、随机数、序列生成、特殊矩阵
- **设备与类型推断**: 如何确定输出 Tensor 的 dtype 和 device

**关键源码位置**:
```
torch/_refs/__init__.py       # 工厂函数引用实现
torch/_ops.py                 # 算子注册系统
torch/library.py              # Library 类
```

**代码示例**:
```python
# 工厂函数调用链
torch.zeros(3, 4)
    ↓
torch.ops.aten.zeros.default([3, 4])
    ↓
C++ ATen 实现
```

**适合读者**: 想了解算子创建和注册机制的读者

---

### 06. Dispatcher 调度系统 (`06-dispatcher.md`)

**本文档解答的问题**:
- PyTorch 如何将算子路由到 CPU/CUDA 等不同后端？
- Dispatch Key 是什么？
- 如何实现自定义算子？

**核心内容**:
- **Dispatch Key 机制**: 算子路由的核心数据结构
- **Key 层次结构**: Autograd → Device → Backend → Composite
- **算子注册系统**: `torch.library.impl` 注册不同后端的实现
- **`__torch_dispatch__`**: Python 层的调度拦截协议
- **自定义算子实战**: 从零定义并实现自定义算子

**关键源码位置**:
```
torch/library.py              # 算子注册 API
torch/_C/__init__.pyi.in      # DispatchKey 枚举
torch/_dispatch/python.py     # Python Dispatcher
```

**Dispatch 流程**:
```
用户调用 torch.add(a, b)
         ↓
   Dispatcher 检查 Key 队列
         ↓
[Autograd] → [CUDA] → [CPU] → [Composite]
         ↓
   执行匹配的 Kernel
```

**适合读者**: 想了解 PyTorch 扩展机制的进阶读者

---

## 学习路线建议

### 入门路径 (1-2 周)
```
第 1 步：01-architecture.md (整体架构)
        理解 Tensor 的基本结构和继承关系
        
        ↓
        
第 2 步：02-tensor-operations.md (基本操作)
        了解常用操作的实现方式
        
        ↓
        
第 3 步：04-storage-memory.md (底层存储)
        理解 Tensor 数据如何在内存中存储
```

### 进阶路径 (2-4 周)
```
第 4 步：03-autograd.md (自动微分)
        深入理解梯度计算原理
        
        ↓
        
第 5 步：05-factory-functions.md (工厂函数)
        了解算子创建机制
        
        ↓
        
第 6 步：06-dispatcher.md (调度系统)
        掌握 PyTorch 扩展机制
```

### 实践项目
完成学习后，尝试以下项目：
- 自定义 Tensor 子类（如带日志的 Tensor）
- 实现自定义 autograd Function
- 注册自定义算子并支持 CPU/CUDA 后端

---

## 源码定位工具

### 搜索命令速查

```bash
# 定位类定义
rg "class Tensor" torch/_tensor.py
rg "class _StorageBase" torch/storage.py
rg "class Function" torch/autograd/function.py

# 定位方法定义
rg "def backward" torch/_tensor.py
rg "def __torch_dispatch__" torch/

# 搜索 DispatchKey
rg "DispatchKey\." torch/library.py

# 查看文件结构
tree -L 2 torch/autograd/
```

### 关键文件索引

| 模块 | 文件路径 | 内容 |
|------|----------|------|
| Tensor | `torch/_tensor.py` | Tensor 类定义 (L110+) |
| Storage | `torch/storage.py` | 存储系统 (_StorageBase: L41) |
| Autograd | `torch/autograd/` | 自动微分 (function.py, grad_mode.py) |
| Dispatcher | `torch/library.py` | 算子注册 (Library 类) |
| 工厂函数 | `torch/_refs/__init__.py` | 引用实现 |
| 重载协议 | `torch/overrides.py` | `__torch_function__` |
| C++ 绑定 | `torch/_C/__init__.pyi.in` | 类型注解 |

---

## 调试与实验

### 启用调试日志

```bash
# Dispatch 日志 - 查看算子如何被调度
TORCH_LOGS=dispatch python script.py

# 算子 schema 日志
TORCH_LOGS=op_schema python script.py

# 梯度检查 - 检测 NaN/Inf
torch.autograd.set_detect_anomaly(True)
```

### 检查 Dispatch 状态

```python
import torch
from torch.library import has_kernel, get_kernel

# 查看算子实现
print(torch.ops.aten.add.default)

# 检查 Kernel 是否注册
print(has_kernel("aten::add.Tensor", "CPU"))    # True
print(has_kernel("aten::add.Tensor", "CUDA"))   # True (如有 GPU)

# 获取特定实现
cpu_impl = get_kernel("aten::add.Tensor", "CPU")
```

### 使用 Tensor 子类调试

```python
class DebugTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        print(f"Calling {func.__name__}")
        return super().__torch_function__(func, types, args, kwargs)

x = DebugTensor([1.0, 2.0, 3.0])
y = x + 1  # 打印：Calling add.Tensor
```

---

## 后续扩展

计划中的后续文档：

- [ ] 07. Tensor 子类实战 - 完整指南：从继承到序列化
- [ ] 08. 设备管理 - 设备抽象、跨设备操作、自定义设备
- [ ] 09. 序列化深入 - pickle、torch.save/load、跨版本兼容
- [ ] 10. 性能分析工具 - 内存分析、算子耗时、瓶颈定位

---

## 常见问题 (FAQ)

### Q: 为什么我的修改没有生效？
A: PyTorch 是 C++ 扩展，纯 Python 修改可能需要重新安装：`pip install -e . -v --no-build-isolation`

### Q: 行号对不上怎么办？
A: 文档基于 PyTorch v2.11.0，不同版本行号可能有差异。使用 `rg` 搜索类名/方法名定位

### Q: 如何贡献代码？
A: 参考 [PyTorch 贡献指南](https://pytorch.org/docs/stable/community/contribution_guide.html)

---

## 参考资料

### 官方资源
- [PyTorch 官方文档](https://pytorch.org/docs/stable/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [PyTorch 贡献指南](https://pytorch.org/docs/stable/community/contribution_guide.html)

### 相关源码目录
```
torch/_tensor.py              # Tensor 类
torch/storage.py              # Storage 系统
torch/autograd/               # 自动微分
torch/library.py              # 算子注册
torch/_refs/__init__.py       # 引用实现
torch/overrides.py            # __torch_function__
torch/csrc/autograd/          # C++ 自动微分引擎
```

### 外部学习资源
- [PyTorch Internals](https://pytorch.org/docs/stable/community/design.html) - 设计文档
- [PyTorch 源码解读系列](https://zhuanlan.zhihu.com/pytorch-source) - 中文社区解读
