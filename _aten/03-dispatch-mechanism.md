# ATen - 核心算子库（三）：Dispatch 机制详解

> **前序**: [Part 2 - 算子声明系统](./02-native-functions-yaml.md)  
> **核心源码**: `aten/src/ATen/core/dispatch/Dispatcher.h`, `c10/core/DispatchKey.h`

---

## 1. 什么是 Dispatch 机制？

**Dispatch (分发)** 是 PyTorch 的核心机制，用于在运行时根据 Tensor 的属性选择正确的 Kernel 实现。

### 1.1 为什么需要 Dispatch？

一个 `torch.add` 操作，底层可能有几十种实现：

```
add 算子
├── CPU 后端
│   ├── float32 (AVX2 优化)
│   ├── float32 (AVX512 优化)
│   ├── float64
│   ├── int32
│   └── int64
├── CUDA 后端
│   ├── float32
│   ├── float16
│   └── bfloat16
├── MPS 后端
└── ...
```

**Dispatch 的任务**: 根据输入 Tensor 的 `device` 和 `dtype`，选择最优实现。

### 1.2 调用流程

```
Python: torch.add(a, b)
    ↓
Python 绑定：at::add(a, b)
    ↓
Dispatcher: 查找对应 DispatchKey 的 Kernel
    ↓
Kernel: CPU / CUDA / MPS 实现
    ↓
计算结果
```

---

## 2. DispatchKey: 分发键

### 2.1 DispatchKey 是什么？

**DispatchKey** 是一个枚举类型，标识不同的"后端"或"功能层级"。

**源码**: `c10/core/DispatchKey.h` (L136-L400+)

```cpp
enum class DispatchKey : uint16_t {
    // ~~~~~~~~~~~ 未定义 ~~~~~~~~~~~
    Undefined = 0,      // 空值
    
    // ~~~~~~~~~~~ 功能层级 ~~~~~~~~~~~
    Dense,              // 稠密张量
    Quantized,          // 量化张量
    Sparse,             // 稀疏张量
    SparseCsr,          // 稀疏 CSR 格式
    NestedTensor,       // NestedTensor
    
    // ~~~~~~~~~~~ 后端 (BackendComponent) ~~~~~~~~~~~
    CPU,                // CPU
    CUDA,               // NVIDIA GPU
    HIP,                // AMD GPU
    XLA,                // TPU
    MPS,                // Apple Silicon
    IPU,                // Graphcore IPU
    XPU,                // Intel GPU
    HPU,                // Intel Gaudi
    VE,                 // NEC Vector Engine
    MTIA,               // Meta TPU
    PrivateUse1,        // 私有后端 1
    PrivateUse2,        // 私有后端 2
    PrivateUse3,        // 私有后端 3
    Meta,               // Meta 设备 (无计算)
    
    // ~~~~~~~~~~~ 自动微分 ~~~~~~~~~~~
    AutogradOther,      // 默认 autograd
    AutogradCPU,        // CPU autograd
    AutogradCUDA,       // CUDA autograd
    AutogradXLA,        // XLA autograd
    AutogradNestedTensor,
    
    // ~~~~~~~~~~~ 包装器 ~~~~~~~~~~~
    Tracer,             // 跟踪模式
    AutocastCPU,        // 自动混合精度
    AutocastCUDA,
    FuncTorchBatched,   // vmap
    Batched,
    
    // ... 更多
};
```

### 2.2 DispatchKey 分类

根据 `Note [DispatchKey Classification]` (L114-132):

| 类别 | 说明 | 示例 |
|------|------|------|
| **后端组件** | 标识物理后端 | CPU, CUDA, MPS |
| **功能层级** | 标识张量类型 | Dense, Sparse, Quantized |
| **每后端功能** | 后端×功能的组合 | SparseCPU, AutogradCPU |
| **别名 Key** | 映射到多个 Key | CompositeImplicitAutograd |

### 2.3 DispatchKeySet

多个 DispatchKey 的组合：

```cpp
class DispatchKeySet {
    uint64_t bits_;  // 位图表示
    
public:
    // 添加 Key
    void add(DispatchKey key);
    
    // 获取最高优先级的 Key
    DispatchKey highestPriorityKey() const;
};
```

**位图设计**: 每个 Key 对应一个 bit，便于快速运算。

---

## 3. Dispatcher: 分发器

### 3.1 单例模式

**源码**: `Dispatcher.h` (L71-L132)

```cpp
class Dispatcher final {
public:
    // 全局单例访问
    static Dispatcher& singleton() {
        static Dispatcher& s = realSingleton();
        return s;
    }
    
private:
    Dispatcher();  // 私有构造函数
    
    // 算子注册表
    std::list<OperatorDef> operators_;
    LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
};
```

**设计要点**:
- 单例模式保证全局唯一的注册表
- `LeftRight` 锁保证并发安全

### 3.2 核心数据结构

**OperatorDef** (L76-L93):
```cpp
struct OperatorDef {
    explicit OperatorDef(OperatorName&& op_name) : op(std::move(op_name)) {}
    
    impl::OperatorEntry op;  // 算子条目
    size_t def_count = 0;     // schema 注册计数
    size_t def_and_impl_count = 0;  // schema+kernel 注册计数
};
```

**OperatorHandle** (L443-L550+):
```cpp
class OperatorHandle {
public:
    const OperatorName& operator_name() const;
    const FunctionSchema& schema() const;
    
    // 查找特定 DispatchKey 的 Kernel
    bool hasKernelForDispatchKey(DispatchKey k) const;
    SafeKernelFunction getComputedKernelForDispatchKey(DispatchKey k) const;
    
private:
    OperatorDef* operatorDef_;
};
```

---

## 4. 算子注册流程

### 4.1 注册时机

算子注册发生在**程序启动时的静态初始化阶段**。

### 4.2 注册方式

#### 方式 1: YAML 自动生成 (主要方式)

```yaml
# native_functions.yaml
- func: abs(Tensor self) -> Tensor
  dispatch:
    CPU: abs_cpu
    CUDA: abs_cuda
```

生成的 C++ 注册代码：
```cpp
// 生成的 RegisterSchema.cpp
namespace {
    TORCH_ATTRIBUTE_UNUSED auto registry = register_ops([](OperatorHandle& op) {
        op.impl("CPU", &abs_cpu);
        op.impl("CUDA", &abs_cuda);
    });
}
```

#### 方式 2: 手动注册

```cpp
// 使用 RegisterOperators API
#include <ATen/core/op_registration/op_registration.h>

namespace {
    class abs_kernel_cpu final : public c10::OperatorKernel {
    public:
        Tensor operator()(const Tensor& self) {
            // 实现
        }
    };
}

static auto registry = c10::RegisterOperators()
    .op(c10::RegisterOperators::options()
        .schema("aten::abs(Tensor self) -> Tensor")
        .kernel<abs_kernel_cpu>(DispatchKey::CPU));
```

### 4.3 注册流程

```
1. 解析 native_functions.yaml
       ↓
2. 生成 RegisterSchema.cpp
       ↓
3. 静态初始化时调用 register_def()
       ↓
4. 创建 OperatorHandle
       ↓
5. 调用 register_impl() 注册 Kernel
       ↓
6. 填充 DispatchTable
```

---

## 5. 算子查找与调用

### 5.1 查找流程

**源码**: `Dispatcher.h` (L145-L163)

```cpp
// 查找算子
std::optional<OperatorHandle> findSchema(const OperatorName& operator_name);

// 查找或抛出异常
OperatorHandle findSchemaOrThrow(const char* name, const char* overload_name);
```

### 5.2 调用流程

**源码**: `Dispatcher.h` (L178-L201)

```cpp
template <class Return, class... Args>
Return call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
    // 1. 从参数中提取 DispatchKeySet
    DispatchKeySet dispatchKeySet = computeDispatchKeySet(args...);
    
    // 2. 查找最高优先级的 Kernel
    const KernelFunction& kernel = op.getComputedKernelForDispatchKey(
        dispatchKeySet.highestPriorityKey()
    );
    
    // 3. 调用 Kernel
    return kernel.call<Return, Args...>(args...);
}
```

### 5.3 调用示例

```cpp
// 用户代码
Tensor c = at::add(a, b);

// Dispatcher 内部流程
Tensor add(const Tensor& a, const Tensor& b) {
    // 1. 获取算子 Handle
    static auto op = Dispatcher::singleton().findSchemaOrThrow("aten::add", "");
    
    // 2. 计算 DispatchKeySet (从参数 a, b)
    DispatchKeySet keys = computeDispatchKeySet(a, b);
    // 如果 a 是 CPU Tensor, keys = {CPU, Dense, ...}
    
    // 3. 获取 Kernel
    auto kernel = op.getComputedKernelForDispatchKey(keys.highestPriorityKey());
    
    // 4. 调用
    return kernel.call<Tensor, const Tensor&, const Tensor&>(a, b);
}
```

---

## 6. DispatchKey 计算

### 6.1 DispatchKeyExtractor

**源码**: `aten/src/ATen/core/dispatch/DispatchKeyExtractor.h`

```cpp
class DispatchKeyExtractor {
public:
    // 从函数参数提取 DispatchKeySet
    DispatchKeySet getDispatchKeySet(const Stack& args) const;
    
private:
    // 记录哪些参数是 Tensor
    std::vector<int> tensor_arg_indices_;
};
```

### 6.2 计算逻辑

```
输入：Tensor a (CPU, float32), Tensor b (CPU, float32)

1. 提取 Tensor 参数: a, b
       ↓
2. 获取每个 Tensor 的 DispatchKey
   - a.key_set() = {CPU, Dense}
   - b.key_set() = {CPU, Dense}
       ↓
3. 合并 KeySet
   - union = {CPU, Dense}
       ↓
4. 应用 TLS 排除
   - 如果 TLS 排除了 CPU，则移除
       ↓
5. 输出：{CPU, Dense}
```

### 6.3 特殊情况

**多 Tensor 不同设备**:
```python
a = torch.tensor([1.0], device='cpu')
b = torch.tensor([2.0], device='cuda')
c = a + b  # 错误！设备不匹配
```

**Scalar 参数**:
```python
# Scalar 不影响 DispatchKey
c = a + 1.0  # DispatchKey 由 a 决定
```

---

## 7. Composite 后端

### 7.1 什么是 Composite？

**Composite** 是一种"元后端"，实现一次，所有后端通用。

### 7.2 两种类型

#### CompositeImplicitAutograd

```cpp
// 实现：调用其他 at:: 算子
Tensor my_op(const Tensor& self) {
    return self * 2 + 1;  // 调用 mul 和 add
}
```

**特点**:
- 自动支持 autograd
- 适用于所有后端
- **默认注册目标** (YAML 不指定 dispatch 时)

#### CompositeExplicitAutograd

```cpp
// 实现：显式计算，不调用其他 at:: 算子
Tensor abs(const Tensor& self) {
    // 直接操作数据
    return self.data_ptr() > 0 ? self : -self;
}
```

**特点**:
- 不自动支持 autograd
- 需要在 `derivatives.yaml` 定义 backward
- 适用于所有后端

### 7.3 选择决策树

```
你的 Kernel 是否调用其他 at:: 算子？
├── Yes → 是否自动支持 autograd？
│         ├── Yes → CompositeImplicitAutograd (或不指定)
│         └── No → CompositeExplicitAutograd
│
└── No (直接操作数据) → 使用具体后端 (CPU/CUDA)
```

---

## 8. BackendFallback: 后端回退

### 8.1 什么是 Fallback？

当某个 DispatchKey 没有注册 Kernel 时，Fallback 提供兜底实现。

### 8.2 注册 Fallback

```cpp
// 为 CPU 注册 fallback
static auto fallback = c10::RegisterOperators()
    .op([](const Tensor& self) {
        // 默认实现：抛出错误
        TORCH_CHECK(false, "No CPU implementation");
    }, c10::DispatchKey::CPU);
```

### 8.3 实际用途

**Out-of-tree 后端** (如 XLA):
```cpp
// XLA 只实现部分算子，其他回退到 CPU
c10::RegisterOperators()
    .op("xla_specific_op", &xla_kernel, DispatchKey::XLA)
    .fallback(&cpu_fallback, DispatchKey::XLA);
```

---

## 9. 实战：追踪 Dispatch

### 9.1 启用调试输出

```python
import torch

# 方法 1: 显示 Dispatch 过程
torch._C._log_api_usage_once("dispatch_trace")

# 方法 2: 使用 PythonDispatcher
from torch._python_dispatcher import PythonDispatcher

dispatcher = PythonDispatcher()
dispatcher.register(["CPU", "CUDA", "CompositeImplicitAutograd"])

# 查看 dispatch 表
print(dispatcher.dispatchTable("aten::abs"))
```

### 9.2 查看注册表

```cpp
// C++ 代码
auto& dispatcher = c10::Dispatcher::singleton();
auto names = dispatcher.getRegistrationsForDispatchKey(c10::DispatchKey::CPU);

for (const auto& name : names) {
    std::cout << name.operator_name() << std::endl;
}
```

### 9.3 常见错误排查

**错误 1**: "Could not find kernel for dispatch key CPU"

```
原因：算子没有在 CPU 后端注册
解决：检查 native_functions.yaml 的 dispatch 字段
```

**错误 2**: "Tried to register operator twice"

```
原因：静态初始化顺序问题导致重复注册
解决：使用匿名 namespace 包裹 registry
```

**错误 3**: "Expected all tensors to be on the same device"

```
原因：多 Tensor 参数设备不一致
解决：在 YAML 添加 device_check: NoCheck (如果允许)
```

---

## 10. 高级主题

### 10.1 TLS (Thread Local State) 排除

```cpp
// 在 Kernel 内部排除某些 DispatchKey
{
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::my_op(...);  // 跳过某些层级
}
```

### 10.2 Redispatch

```cpp
// Kernel 内部调用 Dispatcher
Tensor my_kernel(const Tensor& self) {
    // 重新 dispatch (用于调用其他后端)
    return at::redispatch::my_op(
        DispatchKeySet(DispatchKey::Dense),  // 降级 KeySet
        self
    );
}
```

### 10.3 自定义 DispatchKey

```cpp
// 扩展 DispatchKey (out-of-tree)
enum MyDispatchKey {
    MyBackend = DispatchKey::PrivateUse1,
};

// 注册 Kernel
c10::RegisterOperators()
    .op("my_op", &my_kernel, MyDispatchKey);
```

---

## 11. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `c10/core/DispatchKey.h` | L36-L103 | BackendComponent 枚举 |
| `c10/core/DispatchKey.h` | L136-L400+ | DispatchKey 完整枚举 |
| `c10/core/DispatchKey.h` | L114-L132 | Note [DispatchKey Classification] |
| `Dispatcher.h` | L71-L132 | Dispatcher 单例定义 |
| `Dispatcher.h` | L145-L163 | 查找接口 |
| `Dispatcher.h` | L178-L201 | 调用接口 |
| `Dispatcher.h` | L443-L550 | OperatorHandle |
| `op_registration.h` | L53-L150 | RegisterOperators API |
| `DispatchStub.h` | L10-L50 | CPU 指令集分发 |

---

## 12. 下一步

| Part | 主题 |
|------|------|
| [Part 4](./04-kernel-implementation.md) | C++ Kernel 实现 |
| [Part 5](./05-registration.md) | 注册机制详解 |

---

**参考资料**:
- `c10/core/DispatchKey.h` - DispatchKey 完整定义
- `aten/src/ATen/core/dispatch/Dispatcher.h` - Dispatcher 实现
- `aten/src/ATen/core/op_registration/` - 注册 API
- [Note: Per-Backend Functionality Dispatch Keys](c10/core/DispatchKey.h)
