# ATen - 核心算子库（五）：注册机制详解

> **前序**: [Part 4 - C++ Kernel 实现](./04-kernel-implementation.md)  
> **核心源码**: `aten/src/ATen/core/op_registration/op_registration.h`, `aten/src/ATen/native/DispatchStub.h`

---

## 1. 注册机制概览

在 Part 4 中，我们了解了 Kernel 的实现方式。本 Part 深入探讨这些 Kernel 是如何注册到 Dispatcher 的。

### 1.1 两种注册方式

PyTorch 提供两种算子注册方式：

| 方式 | 适用场景 | 注册位置 |
|------|---------|---------|
| **YAML 自动生成** | 绝大多数算子 | `native_functions.yaml` |
| **手动注册** | 特殊算子 / 自定义算子 | C++ 代码中使用 `RegisterOperators` |

### 1.2 注册时机

**静态初始化阶段**:
```cpp
// 静态全局对象，在 main() 之前执行
static auto registry = c10::RegisterOperators()
    .op("my_op", &my_kernel, DispatchKey::CPU);

// 或在匿名 namespace 中
namespace {
    auto _ = []() {
        // 注册逻辑
        return 0;
    }();
}
```

---

## 2. YAML 自动生成注册

### 2.1 从 YAML 到 C++ 注册代码

**YAML 声明** (`native_functions.yaml`):
```yaml
- func: abs(Tensor self) -> Tensor
  variants: function, method
  dispatch:
    CPU: abs_cpu
    CUDA: abs_cuda
```

**生成的注册代码** (`RegisterSchema.cpp`):
```cpp
// 工具生成的注册代码
namespace {
    TORCH_ATTRIBUTE_UNUSED auto registry = register_ops([](OperatorHandle& op) {
        op.impl("CPU", &abs_cpu);
        op.impl("CUDA", &abs_cuda);
    });
}
```

### 2.2 生成的注册流程

```
1. 解析 native_functions.yaml
       ↓
2. 生成 RegisterSchema.cpp
       ↓
3. 编译时包含到 build 中
       ↓
4. 静态初始化时执行注册
       ↓
5. 填充 Dispatcher 注册表
```

### 2.3 默认注册行为

**不指定 dispatch 字段**:
```yaml
# 默认注册到 CompositeImplicitAutograd
- func: my_op(Tensor self) -> Tensor
  # 等价于：
  # dispatch:
  #   CompositeImplicitAutograd: my_op
```

**为什么是 CompositeImplicitAutograd?**
- 适用于所有后端
- 自动支持 autograd
- 实现调用其他 `at::` 算子即可

---

## 3. 手动注册 API

### 3.1 RegisterOperators 类

**源码**: `aten/src/ATen/core/op_registration/op_registration.h` (L53-L200+)

```cpp
/**
 * 一个 RegisterOperators 实例负责一个或多个算子的注册。
 * 必须保持 RegisterOperators 实例存活，否则会在析构时注销算子。
 */
class TORCH_API RegisterOperators final {
public:
    RegisterOperators() = default;
    ~RegisterOperators() = default;
    
    // 链式 API
    RegisterOperators&& op(Options options);
    
    class Options {
    public:
        // 指定 schema
        Options&& schema(const std::string& schemaOrName);
        
        // 注册 functor kernel
        template<class KernelFunctor>
        Options&& kernel(DispatchKey dispatch_key) &&;
        
        // 注册 catch-all kernel
        template<class KernelFunctor>
        Options&& catchAllKernel() &&;
    };
};
```

### 3.2 注册示例

#### 示例 1: 简单函数注册

```cpp
#include <ATen/core/op_registration/op_registration.h>

namespace {
    Tensor my_add_cpu(const Tensor& a, const Tensor& b) {
        return a + b;
    }
}

// 静态注册
static auto registry = c10::RegisterOperators()
    .op(c10::RegisterOperators::options()
        .schema("my::add(Tensor a, Tensor b) -> Tensor")
        .kernel<decltype(&my_add_cpu), &my_add_cpu>(DispatchKey::CPU));
```

#### 示例 2: Functor 注册

```cpp
namespace {
    class MyAddKernel : public c10::OperatorKernel {
    public:
        Tensor operator()(const Tensor& a, const Tensor& b) {
            return a + b;
        }
    };
}

static auto registry = c10::RegisterOperators()
    .op(c10::RegisterOperators::options()
        .schema("my::add(Tensor a, Tensor b) -> Tensor")
        .kernel<MyAddKernel>(DispatchKey::CPU));
```

#### 示例 3: Catch-All Kernel

```cpp
// Catch-All: 忽略 DispatchKey，总是调用这个 kernel
static auto registry = c10::RegisterOperators()
    .op(c10::RegisterOperators::options()
        .schema("my::custom_op(Tensor x) -> Tensor")
        .catchAllKernel<MyCustomKernel>());
```

### 3.3 Kernel 函数签名

**模板参数推导**:
```cpp
// 方式 1: 函数指针
.kernel<decltype(&func), &func>(DispatchKey::CPU)

// 方式 2: Functor
.kernel<MyFunctor>(DispatchKey::CPU)

// 方式 3: Lambda (需要显式类型)
.kernel<decltype(lambda), lambda>(DispatchKey::CPU)
```

---

## 4. REGISTER_DISPATCH 宏机制

### 4.1 宏的定义

**源码**: `DispatchStub.h` (L389-L467)

```cpp
// DECLARE_DISPATCH: 声明 stub 结构
#define DECLARE_DISPATCH(fn, name)                                                         \
  struct name##_DECLARE_DISPATCH_type : DispatchStub<fn, name##_DECLARE_DISPATCH_type> {   \
    /* ... 构造/析构函数 ... */                                                              \
  };                                                                                       \
  extern TORCH_API struct name##_DECLARE_DISPATCH_type name;

// DEFINE_DISPATCH: 定义 stub 实例
#define DEFINE_DISPATCH(name) struct name##_DECLARE_DISPATCH_type name

// REGISTER_DISPATCH: 注册实现 (根据平台自动选择)
#if defined(__CUDACC__)
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
#elif defined(__OBJC__) && defined(USE_MPS)
#define REGISTER_DISPATCH(name, fn) REGISTER_MPS_DISPATCH(name, fn)
#elif defined(CPU_CAPABILITY)
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif
```

### 4.2 各后端注册宏

```cpp
// CUDA 注册
#define REGISTER_CUDA_DISPATCH(name, fn) \
  static RegisterCUDADispatch<struct name##_DECLARE_DISPATCH_type> \
    name ## __register(name, fn);

// MPS 注册
#define REGISTER_MPS_DISPATCH(name, fn) \
  static RegisterMPSDispatch<struct name##_DECLARE_DISPATCH_type> \
    name ## __register(name, fn);

// CPU 指令集注册
#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
  template <> name##_DECLARE_DISPATCH_type::FnPtr \
  DispatchStub<name##_DECLARE_DISPATCH_type::FnPtr, \
               struct name##_DECLARE_DISPATCH_type>::arch = fn;
```

### 4.3 注册流程图解

```
REGISTER_DISPATCH(add_stub, &add_kernel)
           ↓
展开为 (CUDA 为例):
REGISTER_CUDA_DISPATCH(add_stub, &add_kernel)
           ↓
创建静态对象:
static RegisterCUDADispatch<add_stub_DECLARE_DISPATCH_type> 
    add_stub__register(add_stub, &add_kernel);
           ↓
静态初始化时调用构造函数:
RegisterCUDADispatch(DispatchStub &stub, FnPtr value) {
    stub.set_cuda_dispatch_ptr(value);
}
           ↓
设置函数指针:
impl.cuda_dispatch_ptr = reinterpret_cast<void*>(value);
```

---

## 5. 算子注册表

### 5.1 Dispatcher 注册表结构

**源码**: `Dispatcher.h` (L71-L132)

```cpp
class Dispatcher final {
private:
    // 算子定义列表
    std::list<OperatorDef> operators_;
    
    // 快速查找表 (算子名 → OperatorHandle)
    LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> 
        operatorLookupTable_;
    
    // 单例访问
    static Dispatcher& singleton() {
        static Dispatcher& s = realSingleton();
        return s;
    }
};
```

### 5.2 OperatorDef 结构

```cpp
struct OperatorDef {
    explicit OperatorDef(OperatorName&& op_name) 
        : op(std::move(op_name)) {}
    
    impl::OperatorEntry op;     // 算子条目
    size_t def_count = 0;        // schema 注册计数
    size_t def_and_impl_count = 0;  // schema+kernel 注册计数
};
```

### 5.3 OperatorHandle

**源码**: `Dispatcher.h` (L443-L550)

```cpp
class OperatorHandle {
public:
    // 获取算子名
    const OperatorName& operator_name() const;
    
    // 获取 schema
    const FunctionSchema& schema() const;
    
    // 检查是否有某 DispatchKey 的 kernel
    bool hasKernelForDispatchKey(DispatchKey k) const;
    
    // 获取某 DispatchKey 的 kernel
    SafeKernelFunction getComputedKernelForDispatchKey(DispatchKey k) const;
    
private:
    OperatorDef* operatorDef_;
};
```

---

## 6. 注册流程详解

### 6.1 完整注册流程

```
┌─────────────────────────────────────────────────────────────┐
│                    编译时 (Build Time)                       │
├─────────────────────────────────────────────────────────────┤
│  1. 解析 native_functions.yaml                              │
│  2. 生成 RegisterSchema.cpp                                 │
│  3. 编译为 .o 目标文件                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              静态初始化 (Static Initialization)              │
├─────────────────────────────────────────────────────────────┤
│  4. 全局静态对象构造                                         │
│     - RegisterOperators 实例                                 │
│     - REGISTER_DISPATCH 生成的静态对象                       │
│  5. 调用 impl() / set_*_dispatch_ptr()                      │
│  6. 填充 DispatchTable                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  运行时 (Runtime)                            │
├─────────────────────────────────────────────────────────────┤
│  7. 用户调用 torch.add(a, b)                                │
│  8. Dispatcher::singleton().findSchema(...)                 │
│  9. 查找并调用 Kernel                                       │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 注册代码示例分析

**代码位置**: `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp`

```cpp
// 步骤 1: 在头文件中声明
// BinaryOps.h
using add_fn_type = Tensor(*)(const Tensor&, const Tensor&);
DECLARE_DISPATCH(add_fn_type, add_stub);

// 步骤 2: 在主 cpp 文件中定义
// BinaryOps.cpp
DEFINE_DISPATCH(add_stub);

// 步骤 3: 在 cpu 子目录中注册实现
// cpu/BinaryOpsKernel.cpp
namespace {
    Tensor add_kernel_cpu(const Tensor& a, const Tensor& b) {
        // 实际实现
    }
}

REGISTER_DISPATCH(add_stub, &add_kernel_cpu);
```

---

## 7. Composite 后端注册

### 7.1 CompositeImplicitAutograd

**特点**:
- 不需要在 YAML 中指定 dispatch
- 自动支持所有后端
- 自动支持 autograd

**实现模式**:
```cpp
Tensor my_op(const Tensor& self) {
    // 调用其他 at:: 算子
    return at::relu(at::conv2d(self, weight, bias));
}
```

### 7.2 CompositeExplicitAutograd

**特点**:
- 适用于所有后端
- 需要手动定义 backward (derivatives.yaml)
- 直接操作数据，不调用其他 `at::` 算子

**实现模式**:
```cpp
Tensor abs(const Tensor& self) {
    // 直接计算，不调用其他 at:: 算子
    return at::native::abs_out(self);
}
```

### 7.3 注册优先级

```
CompositeImplicitAutograd < CompositeExplicitAutograd < 具体后端

优先级说明:
- 具体后端 (CPU/CUDA) 优先级最高
- CompositeExplicitAutograd 次之
- CompositeImplicitAutograd 最低 (默认 fallback)
```

---

## 8. 自定义算子注册实战

### 8.1 完整示例

**Step 1: 创建头文件**
```cpp
// my_ops.h
#pragma once
#include <ATen/ATen.h>

namespace my_ops {
    Tensor my_add(const Tensor& a, const Tensor& b);
}
```

**Step 2: 实现 Kernel**
```cpp
// my_ops.cpp
#include "my_ops.h"
#include <ATen/core/op_registration/op_registration.h>

namespace my_ops {

Tensor my_add_impl(const Tensor& a, const Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match");
    return a + b;
}

// 注册
static auto registry = c10::RegisterOperators()
    .op(c10::RegisterOperators::options()
        .schema("my_ops::my_add(Tensor a, Tensor b) -> Tensor")
        .kernel<decltype(&my_add_impl), &my_add_impl>(c10::DispatchKey::CPU));

} // namespace my_ops
```

**Step 3: Python 使用**
```python
import torch
import my_ops  # 导入自定义模块

a = torch.randn(3, 3)
b = torch.randn(3, 3)
c = torch.ops.my_ops.my_add(a, b)
```

### 8.2 使用 PYBIND11

```cpp
// python_binding.cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "my_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_add", &my_ops::my_add, "My add operation");
}
```

---

## 9. 调试注册问题

### 9.1 查看注册表

**PythonDispatcher**:
```python
from torch._python_dispatcher import PythonDispatcher

dispatcher = PythonDispatcher()
dispatcher.register(["CPU", "CUDA", "CompositeImplicitAutograd"])

# 查看算子的 dispatch 表
print(dispatcher.dispatchTable("aten::abs"))

# 输出示例:
# {
#   'CPU': <function abs_cpu at 0x...>,
#   'CUDA': <function abs_cuda at 0x...>,
#   ...
# }
```

### 9.2 常见错误

**错误 1**: "Tried to register operator twice"
```
原因：静态初始化顺序导致重复注册
解决：使用匿名 namespace 包裹 registry
```

**错误 2**: "Could not find kernel for dispatch key X"
```
原因：没有为某后端注册 Kernel
解决：在 YAML 中添加 dispatch 或手动注册
```

**错误 3**: "The function signature is not consistent with the kernel"
```
原因：YAML 声明的签名与 C++ 实现不匹配
解决：检查 YAML 和 C++ 的函数签名
```

### 9.3 调试技巧

```bash
# 1. 查看生成的 RegisterSchema.cpp
cat build/aten/src/ATen/RegisterSchema.cpp | grep "abs"

# 2. 查看静态初始化
nm libtorch.so | grep -i "registry"

# 3. Python 中查看算子
python -c "import torch; print(torch.ops.aten.abs)"
```

---

## 10. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `op_registration.h` | L53-L61 | RegisterOperators 类定义 |
| `op_registration.h` | L63-L120 | Options 内部类 |
| `op_registration.h` | L159-L171 | kernel<Functor> 模板方法 |
| `DispatchStub.h` | L389-L398 | DECLARE_DISPATCH 宏 |
| `DispatchStub.h` | L400 | DEFINE_DISPATCH 宏 |
| `DispatchStub.h` | L448-L467 | 各后端 REGISTER_DISPATCH 宏 |
| `Dispatcher.h` | L71-L132 | Dispatcher 单例定义 |
| `Dispatcher.h` | L443-L550 | OperatorHandle 类 |
| `native/README.md` | L536-L595 | dispatch 关键字选择指南 |

---

## 11. 下一步

| Part | 主题 |
|------|------|
| [Part 6](./06-dispatch-macro.md) | AT_DISPATCH 宏与类型分发 |
| [Part 7](./07-tensor-structure.md) | Tensor 核心数据结构 |

---

**参考资料**:
- `aten/src/ATen/core/op_registration/op_registration.h` - 注册 API 完整定义
- `aten/src/ATen/native/DispatchStub.h` - REGISTER_DISPATCH 宏
- `aten/src/ATen/native/README.md` - 注册指南
- `tools/codegen/` - 代码生成器
