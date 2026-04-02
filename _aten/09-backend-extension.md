# ATen - 核心算子库（九）：后端扩展指南

> **前序**: [Part 8 - 自动微分集成](./08-autograd-integration.md)  
> **核心源码**: `aten/src/ATen/native/mps/`, `c10/core/DispatchKey.h`

---

## 1. PyTorch 后端架构

### 1.1 后端分类

PyTorch 支持多种硬件后端：

| 后端 | DispatchKey | 源码位置 | 适用硬件 |
|------|------------|---------|---------|
| **CPU** | `DispatchKey::CPU` | `aten/src/ATen/native/cpu/` | x86/ARM CPU |
| **CUDA** | `DispatchKey::CUDA` | `aten/src/ATen/native/cuda/` | NVIDIA GPU |
| **MPS** | `DispatchKey::MPS` | `aten/src/ATen/native/mps/` | Apple Silicon GPU |
| **XLA** | `DispatchKey::XLA` | (外部仓库) | TPU |
| **MTIA** | `DispatchKey::MTIA` | `aten/src/ATen/native/mtia/` | Meta TPU |
| **PrivateUse1** | `DispatchKey::PrivateUse1` | 自定义 | 私有后端 |

### 1.2 后端扩展方式

| 方式 | 适用场景 | 难度 |
|------|---------|------|
| **完整后端** | 新硬件平台 | 高 |
| **BackendFallback** | 部分算子回退 | 中 |
| **自定义 Device** | 研究/实验 | 中 |

---

## 2. 添加新后端：MPS 案例分析

### 2.1 MPS 后端简介

**MPS (Metal Performance Shaders)** 是 Apple Silicon GPU 的后端实现：

- 使用 Metal 编程模型
- 源码位置：`aten/src/ATen/native/mps/`
- 文件扩展名：`.mm` (Objective-C++)

### 2.2 目录结构

```
aten/src/ATen/native/mps/
├── operations/                 # 算子实现
│   ├── UnaryKernel.mm          # 一元运算
│   ├── BinaryKernel.mm         # 二元运算
│   ├── Activation.mm           # 激活函数
│   ├── Convolution.mm          # 卷积
│   ├── Linear.mm               # 线性层
│   ├── Indexing.mm             # 索引操作
│   └── ...
├── MPSCommon.mm                # 公共工具
├── MPSProfiler.mm              # 性能分析
└── MPSDevice.mm                # 设备管理
```

### 2.3 注册流程

#### Step 1: 在 YAML 中声明

```yaml
# native_functions.yaml
- func: abs(Tensor self) -> Tensor
  dispatch:
    MPS: abs_mps
```

#### Step 2: 实现 Kernel

```cpp
// UnaryKernel.mm (L18-L22)
#define REGISTER_UNARY_TI_DISPATCH(NAME)                    \
  static void NAME##_kernel_mps(TensorIteratorBase& iter) { \
    lib.exec_unary_kernel(iter, #NAME);                     \
  }                                                         \
  REGISTER_DISPATCH(NAME##_stub, NAME##_kernel_mps)

// 注册具体算子
REGISTER_UNARY_TI_DISPATCH(abs);
REGISTER_UNARY_TI_DISPATCH(sin);
REGISTER_UNARY_TI_DISPATCH(exp);
```

#### Step 3: REGISTER_DISPATCH 宏

**源码**: `DispatchStub.h` (L478-L480)

```cpp
#if defined(__OBJC__) && defined(USE_MPS)
#define REGISTER_DISPATCH(name, fn) REGISTER_MPS_DISPATCH(name, fn)
#endif

#define REGISTER_MPS_DISPATCH(name, fn) \
  static RegisterMPSDispatch<struct name##_DECLARE_DISPATCH_type> \
    name ## __register(name, fn);
```

**RegisterMPSDispatch 类** (L355-L359):

```cpp
template <typename DispatchStub>
struct RegisterMPSDispatch {
  RegisterMPSDispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    stub.set_mps_dispatch_ptr(value);
  }
};
```

---

## 3. BackendFallback 机制

### 3.1 什么是 Fallback？

**BackendFallback** 是当某个 DispatchKey 没有注册 Kernel 时的兜底实现。

### 3.2 使用场景

**场景 1: Out-of-tree 后端**
```
XLA 后端：
- 只实现部分算子（如 matmul, conv）
- 其他算子回退到 CPU 实现
```

**场景 2: 渐进式实现**
```
新后端开发：
1. 先实现常用算子
2. 不常用的回退到 CPU
3. 逐步完善
```

### 3.3 注册 Fallback

```cpp
#include <ATen/core/op_registration/op_registration.h>

// 为某后端注册 fallback
static auto fallback = c10::RegisterOperators()
    .op([](const Tensor& self) {
        // 默认实现：抛出错误或回退
        TORCH_CHECK(false, "No MPS implementation");
    }, c10::DispatchKey::MPS);
```

### 3.4 实际案例：XLA

```cpp
// XLA 的 fallback 实现（简化版）
static auto xla_fallback = c10::RegisterOperators()
    .op("aten::add", &cpu_add_fallback, DispatchKey::XLA)
    .op("aten::mul", &cpu_mul_fallback, DispatchKey::XLA);
```

---

## 4. 必须实现的算子集合

### 4.1 核心算子

新后端至少需要实现以下算子才能运行基本模型：

| 类别 | 算子 |
|------|------|
| **Tensor 工厂** | `empty`, `zeros`, `ones`, `full`, `arange` |
| **二元运算** | `add`, `sub`, `mul`, `div` |
| **一元运算** | `neg`, `abs`, `exp`, `log`, `sqrt`, `sin`, `cos` |
| **归约** | `sum`, `mean`, `max`, `min`, `argmax` |
| **矩阵运算** | `mm`, `bmm`, `addmm`, `conv2d` |
| **索引** | `index`, `index_put`, `gather`, `scatter` |
| **形状** | `view`, `transpose`, `permute`, `reshape` |
| **激活函数** | `relu`, `sigmoid`, `tanh`, `softmax` |

### 4.2 测试验证

```python
# 测试新后端
import torch

# 1. 基本运算测试
a = torch.randn(3, 3, device='new_device')
b = torch.randn(3, 3, device='new_device')
c = a + b  # 测试 add

# 2. 常用算子测试
x = torch.randn(10, 10, device='new_device')
y = torch.relu(x)  # 测试 relu
z = torch.matmul(x, y)  # 测试 matmul

# 3. 反向传播测试
x.requires_grad_(True)
y = x * 2
y.sum().backward()  # 测试 autograd
```

---

## 5. 完整后端实现步骤

### 5.1 步骤概览

```
1. 定义新的 DispatchKey
       ↓
2. 创建后端目录结构
       ↓
3. 实现核心算子
       ↓
4. 注册 Kernel
       ↓
5. 实现 BackendFallback
       ↓
6. 测试与验证
```

### 5.2 详细步骤

#### Step 1: 定义 DispatchKey

**源码**: `c10/core/DispatchKey.h`

```cpp
enum class DispatchKey : uint16_t {
    // ... 现有 Key
    
    // 添加新后端
    MyBackend = 42,  // 选择未使用的值
    
    // 或者使用预留的
    PrivateUse1,  // 用于外部后端
    PrivateUse2,
    PrivateUse3,
};
```

#### Step 2: 创建目录结构

```
aten/src/ATen/native/mybackend/
├── MyBackendCommon.mm      # 公共工具
├── UnaryKernel.mm          # 一元运算
├── BinaryKernel.mm         # 二元运算
├── Activation.mm           # 激活函数
└── ...
```

#### Step 3: 实现 Kernel

```cpp
// MyBackendKernel.mm
#include <ATen/native/DispatchStub.h>
#include <ATen/TensorIterator.h>

namespace at::native {

void mybackend_abs_kernel(TensorIteratorBase& iter) {
    // 使用 Metal/CUDA/自定义 API 实现
    // ...
}

} // namespace at::native
```

#### Step 4: 注册 Kernel

```cpp
// MyBackendKernel.mm (续)
REGISTER_DISPATCH(abs_stub, &mybackend_abs_kernel);
REGISTER_DISPATCH(add_stub, &mybackend_add_kernel);
```

#### Step 5: 配置构建系统

```cmake
# CMakeLists.txt
if(USE_MYBACKEND)
    add_subdirectory(aten/src/ATen/native/mybackend)
endif()
```

#### Step 6: 测试

```cpp
// test_mybackend.cpp
#include <gtest/gtest.h>

TEST(MyBackendTest, AbsTest) {
    auto a = torch::randn({3, 3}, torch::kMyBackend);
    auto b = torch::abs(a);
    // 验证结果
}
```

---

## 6. 私有后端 (PrivateUse1)

### 6.1 使用 PrivateUse1

`PrivateUse1/2/3` 是预留给外部后端的 DispatchKey：

```cpp
// 不需要修改 PyTorch 核心代码
enum MyDispatchKey {
    MyBackend = DispatchKey::PrivateUse1,
};
```

### 6.2 注册自定义 Kernel

```cpp
#include <ATen/core/op_registration/op_registration.h>

// 使用 PrivateUse1
static auto registry = c10::RegisterOperators()
    .op("aten::abs(Tensor self) -> Tensor",
        &my_abs_kernel,
        c10::DispatchKey::PrivateUse1);
```

### 6.3 完整示例

```cpp
// my_extension.cpp
#include <torch/extension.h>
#include <ATen/core/op_registration/op_registration.h>

// 自定义 abs 实现
torch::Tensor my_abs(const torch::Tensor& self) {
    // 自定义实现
    return self.abs();
}

// 注册
static auto registry = c10::RegisterOperators()
    .op("myext::abs(Tensor self) -> Tensor",
        &my_abs,
        c10::DispatchKey::PrivateUse1);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("abs", &my_abs, "Custom abs operation");
}
```

---

## 7. 后端性能优化

### 7.1 Kernel 融合

```cpp
// 差：多个 Kernel 启动
auto t1 = relu(x);      // Kernel 1
auto t2 = t1 + b;       // Kernel 2
auto t3 = t2 * scale;   // Kernel 3

// 好：融合为一个 Kernel
auto t3 = fused_relu_add_mul(x, b, scale);  // Kernel 1
```

### 7.2 异步执行

```cpp
// 使用 Stream 异步执行
auto stream = at::mps::getCurrentStream();
auto t1 = relu(x);
auto t2 = sin(y);
// t1 和 t2 可以并行执行
```

### 7.3 内存优化

```cpp
// 使用内存池
auto allocator = at::mps::getAllocator();
auto storage = at::Storage(
    at::Storage::use_byte_size_t(),
    size_bytes,
    allocator);
```

---

## 8. 调试后端问题

### 8.1 启用调试输出

```python
import os
os.environ['PYTORCH_MPS_DEBUG'] = '1'  # MPS 调试
os.environ['PYTORCH_CUDA_DEBUG'] = '1'  # CUDA 调试

import torch
```

### 8.2 查看 Dispatch 表

```python
from torch._python_dispatcher import PythonDispatcher

dispatcher = PythonDispatcher()
dispatcher.register(["CPU", "CUDA", "MPS", "PrivateUse1"])

# 查看算子的 dispatch 表
print(dispatcher.dispatchTable("aten::abs"))
```

### 8.3 常见错误

**错误 1**: "Could not find kernel for dispatch key X"
```
原因：没有为该后端注册 Kernel
解决：添加 REGISTER_DISPATCH 或实现 fallback
```

**错误 2**: "Expected all tensors to be on the same device"
```
原因：多 Tensor 参数设备不一致
解决：确保输入 Tensor 在同一设备上
```

**错误 3**: "Fallback to CPU"
```
原因：后端没有实现该算子
解决：实现缺失的算子或显式回退
```

---

## 9. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `DispatchStub.h` | L478-L480 | REGISTER_MPS_DISPATCH 宏 |
| `DispatchStub.h` | L355-L359 | RegisterMPSDispatch 类 |
| `UnaryKernel.mm` | L18-L22 | REGISTER_UNARY_TI_DISPATCH 宏 |
| `UnaryKernel.mm` | L54-L85 | MPS Kernel 注册示例 |
| `DispatchKey.h` | L136-L400+ | DispatchKey 完整枚举 |
| `op_registration.h` | L53-L200+ | RegisterOperators API |

---

## 10. 下一步

| Part | 主题 |
|------|------|
| [Part 10](./10-debugging-testing.md) | 调试与测试 |

---

**参考资料**:
- `aten/src/ATen/native/mps/` - MPS 后端完整实现
- `c10/core/DispatchKey.h` - DispatchKey 定义
- `aten/src/ATen/core/op_registration/op_registration.h` - 注册 API
- [PyTorch 后端扩展指南](https://pytorch.org/docs/stable/extend/extending.html)
