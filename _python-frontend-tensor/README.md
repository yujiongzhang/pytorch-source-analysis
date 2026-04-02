# Python 层 Tensor API

本目录包含 PyTorch Python 层 Tensor API 的源码分析文档。

---

## 目录

| 章节 | 主题 | 描述 |
|------|------|------|
| [Part 1](./01-tensor-class-structure.md) | Tensor 类结构 | Python Tensor 类定义、继承关系、C++ 绑定机制 |
| [Part 2](./02-tensor-creation.md) | Tensor 构造与工厂函数 | Tensor 创建机制、数据转换、工厂函数 |
| [Part 3](./03-autograd.md) | 自动微分集成 | autograd 引擎、Function、梯度累积 |
| [Part 4](./04-storage-memory.md) | 存储与内存管理 | Storage 结构、内存分配器、共享内存 |
| [Part 5](./05-factory-functions.md) | 工厂函数详解 | 各类工厂函数实现细节 |
| [Part 6](./06-dispatcher.md) | 分发机制 | `__torch_dispatch__`、`__torch_function__` |

---

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                  Python 层 (torch/)                      │
│  - torch/_tensor.py: Tensor 类定义                       │
│  - torch/_VF.py: 向量函数                                │
│  - torch/functional.py: 函数式 API                        │
│  - torch/autograd/: 自动微分                             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              C++ 扩展层 (torch/csrc/)                    │
│  - tensor/python_tensor.cpp: Tensor 类型绑定              │
│  - autograd/python_engine.cpp: 引擎绑定                   │
│  - Storage.cpp: Storage 绑定                             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              C++ 核心层 (ATen + c10)                     │
│  - aten/src/ATen/: 算子实现                              │
│  - c10/core/: 核心数据结构                               │
└─────────────────────────────────────────────────────────┘
```

---

## 核心概念

### Tensor 层次结构

```
Python Tensor (torch.Tensor)
    ↓ 继承
torch._C.TensorBase (C++ 扩展类)
    ↓ 包装
at::Tensor (C++ 高级 API)
    ↓ 指向
TensorImpl (元数据 + Storage 指针)
    ↓ 指向
Storage (内存管理)
    ↓ 指向
StorageImpl + DataPtr
    ↓
实际数据内存
```

### 关键组件

| 组件 | 文件 | 职责 |
|------|------|------|
| Tensor | `torch/_tensor.py` | Python 层 Tensor 类 |
| TensorBase | `torch/_C` | C++ 扩展基类 |
| THPVariable | `torch/csrc/autograd/python_variable.h` | C++ Variable 结构 |
| Storage | `torch/csrc/Storage.cpp` | Python Storage 绑定 |
| Engine | `torch/csrc/autograd/engine.cpp` | 反向传播引擎 |

---

## 快速开始

### 阅读顺序

1. **入门**: 从 [Part 1](./01-tensor-class-structure.md) 开始，了解 Tensor 类结构
2. **构造**: 阅读 [Part 2](./02-tensor-creation.md)，理解 Tensor 如何创建
3. **微分**: 阅读 [Part 3](./03-autograd.md)，学习自动微分机制
4. **内存**: 阅读 [Part 4](./04-storage-memory.md)，掌握内存管理
5. **工厂函数**: 阅读 [Part 5](./05-factory-functions.md)，了解各类工厂函数
6. **分发**: 阅读 [Part 6](./06-dispatcher.md)，理解分发机制

### 前置知识

- 熟悉 Python 编程
- 了解 C++ 基础
- 理解 PyTorch 基本使用
- 熟悉自动微分概念

---

## 源码位置

### Python 层

```
pytorch/torch/
├── _tensor.py              # Tensor 类定义
├── _VF.py                  # 向量函数
├── _C/
│   └── __init__.pyi.in     # C++ 扩展类型注解
├── autograd/
│   ├── __init__.py         # autograd API
│   ├── function.py         # Function 基类
│   ├── grad_mode.py        # 梯度模式
│   └── graph.py            # 计算图
└── storage.py              # Storage API
```

### C++ 层

```
pytorch/torch/csrc/
├── tensor/
│   ├── python_tensor.h     # Tensor 绑定声明
│   └── python_tensor.cpp   # Tensor 绑定实现
├── autograd/
│   ├── python_variable.h   # Variable 绑定声明
│   ├── python_variable.cpp # Variable 绑定实现
│   ├── python_engine.h     # 引擎绑定声明
│   ├── python_engine.cpp   # 引擎绑定实现
│   └── engine.cpp          # 反向传播引擎
└── Storage.cpp             # Storage 绑定
```

### 核心层

```
pytorch/
├── aten/src/ATen/
│   ├── native/
│   │   └── TensorFactories.cpp  # 工厂函数
│   └── core/
└── c10/core/
    ├── Storage.h           # Storage 定义
    ├── StorageImpl.h       # StorageImpl 定义
    ├── TensorImpl.h        # TensorImpl 定义
    └── impl/
        └── DataPtrImpl.h   # DataPtr 实现
```

---

## 关键 API

### Tensor 构造

```python
# 从数据创建
torch.tensor([1, 2, 3])
torch.as_tensor(array)
torch.from_numpy(ndarray)

# 工厂函数
torch.empty(3, 4)
torch.zeros(3, 4)
torch.ones(3, 4)
torch.arange(0, 10, 2)
torch.linspace(0, 1, 10)

# 基于现有 Tensor
x.new_ones(3, 4)
x.new_zeros(3, 4)
x.new_empty(3, 4)
```

### 自动微分

```python
# 反向传播
loss.backward()
torch.autograd.backward(loss)

# 计算梯度
grads = torch.autograd.grad(loss, params)

# 梯度模式
with torch.no_grad():
    ...

with torch.enable_grad():
    ...

# 异常检测
with torch.autograd.detect_anomaly():
    ...
```

### 内存管理

```python
# Storage
storage = torch.FloatStorage(100)
tensor = torch.FloatTensor(storage)

# 共享内存
storage.share_memory_()

# CUDA 内存
torch.cuda.memory_allocated()
torch.cuda.empty_cache()
```

---

## 后续阅读

- [ATen 层分析](../_aten/README.md): ATen 核心算子库
- [Inductor 层分析](../_inductor/README.md): TorchInductor 编译器

---

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这些文档！

---

**最后更新**: 2026-04-02
