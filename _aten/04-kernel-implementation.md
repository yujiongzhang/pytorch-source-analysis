# ATen - 核心算子库（四）：C++ Kernel 实现

> **前序**: [Part 3 - Dispatch 机制](./03-dispatch-mechanism.md)  
> **核心源码**: `aten/src/ATen/native/DispatchStub.h`, `aten/src/ATen/native/TensorIterator.h`

---

## 1. Kernel 实现概览

在 Part 3 中，我们了解了 Dispatcher 如何根据 DispatchKey 找到对应的 Kernel 函数。本 Part 深入探讨 Kernel 的实际实现。

### 1.1 Kernel 的组织形式

ATen 的 Kernel 实现按功能和后端分类：

```
aten/src/ATen/native/
├── Activation.cpp              # 激活函数 (ReLU, Sigmoid, etc.)
├── BinaryOps.cpp               # 二元运算 (add, mul, div)
├── TensorIterator.cpp          # TensorIterator 实现
├── Loss.cpp                    # 损失函数
├── LinearAlgebra.cpp           # 线性代数
│
├── cpu/                        # CPU 优化实现
│   ├── BinaryOpsKernel.cpp     # 二元运算 CPU Kernel
│   ├── Activation.cpp          # 激活函数 CPU Kernel
│   ├── Loops.h                 # CPU Kernel 辅助函数
│   └── ...
│
├── cuda/                       # CUDA 实现
│   ├── BinaryOpsKernels.cu     # 二元运算 CUDA Kernel
│   ├── Activation.cu           # 激活函数 CUDA Kernel
│   ├── Loops.cuh               # CUDA Kernel 辅助函数
│   └── ...
│
└── mps/                        # MPS (Metal) 实现
    └── operations/
```

### 1.2 实现层次

```
YAML 声明 (native_functions.yaml)
       ↓
函数包装层 (Activation.cpp)     ← 薄的包装函数
       ↓
Kernel 实现层 (cpu/Activation.cpp)  ← 实际计算逻辑
       ↓
硬件指令 (CPU/GPU 计算)
```

---

## 2. DispatchStub: CPU 指令集分发

### 2.1 为什么需要 DispatchStub？

现代 CPU 支持不同的指令集（AVX2, AVX512 等），同一份代码编译一次无法充分利用硬件性能。

**DispatchStub** 允许:
- 同一 Kernel 编译多次，每次使用不同的编译器标志
- 运行时根据 CPU 能力选择最优实现

### 2.2 DispatchStub 结构

**源码**: `aten/src/ATen/native/DispatchStub.h` (L10-L50)

```cpp
// 第 10-36 行：注释说明使用方式
// Example:
// In native/MyKernel.h:
//   using fn_type = void(*)(const Tensor& x);
//   DECLARE_DISPATCH(fn_type, stub)
//
// In native/MyKernel.cpp
//   DEFINE_DISPATCH(stub);
//
// In native/cpu/MyKernel.cpp:
//   namespace {
//     void kernel(const Tensor& x) { ... }
//   }
//   REGISTER_DISPATCH(stub, &kernel);
//
// To call:
//   stub(kCPU, tensor);
```

**核心结构**: `DispatchStubImpl` (L96-L214)

```cpp
struct TORCH_API DispatchStubImpl {
  // 获取指定 device_type 的函数指针
  void* get_call_ptr(c10::DeviceType device_type, ...);
  
  // CPU 指令集选择逻辑
  void* choose_cpu_impl(
    void *DEFAULT    // 默认实现
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void *AVX512   // AVX512 实现
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2     // AVX2 实现
#endif
  );
  
  // 各后端的函数指针存储
  std::atomic<void*> cpu_dispatch_ptr{nullptr};
  void* cuda_dispatch_ptr = nullptr;
  void* mps_dispatch_ptr = nullptr;
  // ...
};
```

### 2.3 CPUCapability 枚举

**源码**: `DispatchStub.h` (L61-L74)

```cpp
enum class CPUCapability {
  DEFAULT = 0,      // 默认实现 (无特殊指令集)
  
#if defined(HAVE_VSX_CPU_DEFINITION)
  VSX = 1,          // IBM PowerPC VSX
#elif defined(HAVE_ZVECTOR_CPU_DEFINITION)
  ZVECTOR = 1,      // IBM Z Vector
#elif defined(HAVE_SVE256_CPU_DEFINITION)
  SVE256 = 1,       // ARM SVE256
#else
  AVX2 = 1,         // Intel AVX2
  AVX512 = 2,       // Intel AVX512
#endif
  
  NUM_OPTIONS
};
```

**编译时检测**:
- 构建时检测 CPU 能力
- 定义 `HAVE_AVX2_CPU_DEFINITION` 等宏
- 条件编译不同版本

### 2.4 使用流程

```
1. 在头文件中声明 stub
   using fn_type = Tensor(*)(const Tensor&);
   DECLARE_DISPATCH(fn_type, relu_stub)
   
2. 在 cpp 文件中定义 stub
   DEFINE_DISPATCH(relu_stub);
   
3. 在 cpu/*.cpp 中注册实现
   REGISTER_DISPATCH(relu_stub, &relu_kernel)
   
4. 调用
   relu_stub(kCPU, tensor)
```

---

## 3. 注册宏详解

### 3.1 REGISTER_DISPATCH 宏

**源码**: `DispatchStub.h` (L468-L492)

```cpp
// 根据编译环境自动选择注册方式
#if defined(__CUDACC__)
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
#elif defined(__HIPCC__)
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
#elif defined(__OBJC__) && defined(USE_MPS)
#define REGISTER_DISPATCH(name, fn) REGISTER_MPS_DISPATCH(name, fn)
#elif defined(CPU_CAPABILITY)
#ifdef CPU_CAPABILITY_AVX512
#define REGISTER_DISPATCH(name, fn) \
  REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#else
#define REGISTER_DISPATCH(name, fn) \
  REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif
#endif
```

### 3.2 各后端注册宏

**源码**: `DispatchStub.h` (L448-L467)

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
#define REGISTER_AVX2_DISPATCH(name, fn) \
  REGISTER_ARCH_DISPATCH(name, AVX2, fn)

// 注册到所有 CPU 指令集
#define REGISTER_ALL_CPU_DISPATCH(name, fn) \
  REGISTER_ARCH_DISPATCH(name, DEFAULT, fn) \
  REGISTER_AVX512_DISPATCH(name, fn) \
  REGISTER_AVX2_DISPATCH(name, fn) \
  // ...
```

### 3.3 注册示例

**示例 1**: 简单 CPU Kernel
```cpp
// 在 BinaryOpsKernel.cpp 中
REGISTER_DISPATCH(add_stub, &add_kernel);
```

**示例 2**: 多指令集注册
```cpp
// 同一 Kernel 注册到多个指令集
REGISTER_ALL_CPU_DISPATCH(mul_stub, &mul_kernel);
```

**示例 3**: 不同指令集不同实现
```cpp
// AVX2 版本
REGISTER_AVX2_DISPATCH(relu_stub, &relu_avx2_kernel);

// 默认版本
REGISTER_ARCH_DISPATCH(relu_stub, DEFAULT, &relu_default_kernel);
```

---

## 4. TensorIterator: 迭代器工具类

### 4.1 什么是 TensorIterator？

**TensorIterator** 是 PyTorch 为逐元素运算设计的迭代器工具，类似 NumPy 的 Array Iterator。

**核心功能**:
1. **Broadcasting**: 自动处理不同形状 Tensor 的广播
2. **类型转换**: 自动处理 dtype 提升和转换
3. **并行化**: 自动将迭代分割到多个线程
4. **矢量化**: 支持 SIMD 指令集优化

**源码位置**: `aten/src/ATen/TensorIterator.h`

### 4.2 TensorIterator 配置

**源码**: `TensorIterator.h` (L32-L66)

```cpp
// 配置示例
auto iter = TensorIteratorConfig()
    .add_output(output)      // 先添加输出
    .add_input(input1)       // 再添加输入
    .add_input(input2)
    .build();
```

**重要规则** (Note [Order of Construction], L48-L55):
- 必须先添加所有输出 Tensor
- 再添加输入 Tensor
- 添加输入后再添加输出会抛出异常

### 4.3 核心数据结构

**OperandInfo** (L117-L200):

```cpp
struct TORCH_API OperandInfo {
  // 数据指针 (迭代器分割后可能与 tensor.data_ptr() 不同)
  void* data = nullptr;
  
  // 广播后的 stride (以字节为单位)
  StrideVector stride_bytes;
  
  // 期望的 dtype 和 device
  std::optional<Device> device;
  ScalarType target_dtype = ScalarType::Undefined;
  ScalarType current_dtype = ScalarType::Undefined;  // 缓存
  
  // 标记是否为输出
  bool is_output = false;
  
  // 标记是否会 resize
  bool will_resize = false;
  
  // 标记是否为读写 (inplace)
  bool is_read_write = false;
};
```

### 4.4 常见 Kernel 模式

---

## 5. 逐元素运算 (Pointwise)

### 5.1 二元运算示例：add

**源码**: `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp`

```cpp
// 第 42-67 行：add_clamp_cpu 函数
void add_clamp_cpu(
    TensorIteratorBase& iter,
    const Scalar& alpha,
    const Scalar& min_val,
    const Scalar& max_val) {
  
  // AT_DISPATCH_ALL_TYPES: 对所有支持的 dtype 生成代码
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "add_clamp_cpu", [&]() {
    auto alpha = alpha_scalar.to<scalar_t>();
    auto alpha_vec = Vectorized<scalar_t>(alpha);
    auto min_scalar = min_val.to<scalar_t>();
    auto min_vec = Vectorized<scalar_t>(min_scalar);
    auto max_scalar = max_val.to<scalar_t>();
    auto max_vec = Vectorized<scalar_t>(max_scalar);
    
    // cpu_kernel_vec: 支持矢量化的 Kernel
    cpu_kernel_vec(
        iter,
        // 标量版本 (非矢量化)
        [=](scalar_t a, scalar_t b) __ubsan_ignore_undefined__ -> scalar_t {
          return std::min(
              max_scalar,
              std::max(min_scalar, static_cast<scalar_t>(a + alpha * b)));
        },
        // 矢量化版本 (SIMD)
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
          return vec_min(
              max_vec,
              vec_max(min_vec, a + alpha_vec * b));
        });
  });
}
```

**关键点**:
1. `AT_DISPATCH_ALL_TYPES` 展开为 switch-case 语句
2. `scalar_t` 被定义为当前 dtype 对应的 C++ 类型
3. `cpu_kernel_vec` 同时提供标量和矢量化版本

### 5.2 浮点运算：atan2

**源码**: `BinaryOpsKernel.cpp` (L69-L85)

```cpp
void atan2_kernel(TensorIteratorBase& iter) {
  // AT_DISPATCH_FLOATING_TYPES_AND2: 包含 float16 和 bfloat16
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "atan2_cpu", [&]() {
    
    cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          return std::atan2(a, b);
        },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
          // 矢量化 atan2 实现
          return Vectorized<scalar_t>::atan2(a, b);
        });
  });
}
```

### 5.3 整数除法

**源码**: `BinaryOpsKernel.cpp` (L208-L215)

```cpp
// 整数除法 (截断模式)
AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_trunc_cpu", [&]() {
  cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
    TORCH_CHECK(b != 0, "ZeroDivisionError");
    return a / b;
  });
});
```

**为什么不用 `cpu_kernel_vec`**?
- 整数除法没有高效的 SIMD 实现
- 对于某些运算，标量版本更简单

### 5.4 位运算

**源码**: `BinaryOpsKernel.cpp` (L413-L422)

```cpp
void bitwise_and_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    // Bool 类型使用逻辑与
    cpu_kernel(iter, [](bool a, bool b) { return a && b; });
  } else {
    // 整数类型使用位与
    _AT_DISPATCH_INTEGRAL_TYPES_V2(iter.dtype(), "bitwise_and_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t { return a & b; },
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { 
            return a & b; 
          });
    });
  }
}
```

---

## 6. 常见 Kernel 辅助函数

### 6.1 cpu_kernel / cpu_kernel_vec

**源码**: `aten/src/ATen/native/cpu/Loops.h`

```cpp
// cpu_kernel: 标量版本
template <typename F>
void cpu_kernel(TensorIteratorBase& iter, F&& f);

// cpu_kernel_vec: 支持矢量化
template <typename F>
void cpu_kernel_vec(
    TensorIteratorBase& iter,
    F&& scalar_f,      // 标量函数
    F&& vec_f          // 矢量化函数
);
```

### 6.2 Vectorized 类型

`Vectorized<scalar_t>` 是 SIMD 向量的包装类：

```cpp
// AVX2 下，Vectorized<float> 包含 8 个 float (256 位)
// AVX512 下，Vectorized<float> 包含 16 个 float (512 位)

// 使用示例
Vectorized<float> a = Vectorized<float>::load(ptr);
Vectorized<float> b = a * 2.0f;  // SIMD 乘法
b.store(ptr);
```

### 6.3 并行化

TensorIterator 自动处理并行化：

```cpp
// 当迭代元素超过 GRAIN_SIZE 时自动并行
// GRAIN_SIZE = 32768 (TensorIterator.h L78)

iter.for_each([=](char** data, int64_t n) {
  // 这个循环可能被分割到多个线程执行
});
```

---

## 7. AT_DISPATCH 宏家族

### 7.1 宏分类

**源码**: `aten/src/ATen/Dispatch.h` (L96-L200+)

| 宏名称 | 支持的 dtype |
|--------|-------------|
| `AT_DISPATCH_ALL_TYPES` | float, double, int32, int64, int16, int8, uint8 |
| `AT_DISPATCH_FLOATING_TYPES` | float, double |
| `AT_DISPATCH_INTEGRAL_TYPES` | int32, int64, int16, int8, uint8 |
| `AT_DISPATCH_ALL_TYPES_AND_HALF` | 上述 + float16 |
| `AT_DISPATCH_FLOATING_TYPES_AND_HALF` | float, double, float16 |

### 7.2 AT_DISPATCH_FLOATING_TYPES 定义

**源码**: `Dispatch.h` (L191-L196)

```cpp
#define AT_DISPATCH_CASE_FLOATING_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
```

### 7.3 AT_DISPATCH_SWITCH 展开

```cpp
AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "atan2_cpu", [&]() {
  // 你的代码
});

// 展开后类似：
switch (iter.dtype()) {
  case ScalarType::Double: {
    using scalar_t = double;
    // 你的代码
    break;
  }
  case ScalarType::Float: {
    using scalar_t = float;
    // 你的代码
    break;
  }
  default:
    TORCH_CHECK(false, "Unsupported dtype");
}
```

---

## 8. 实战：完整的 Kernel 实现

### 8.1 示例：ReLU 实现

**Step 1: YAML 声明**
```yaml
# native_functions.yaml
- func: relu(Tensor self) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: relu
```

**Step 2: 头文件声明**
```cpp
// Activation.h
using relu_fn_type = Tensor(*)(const Tensor&);
DECLARE_DISPATCH(relu_fn_type, relu_stub);
```

**Step 3: Stub 定义**
```cpp
// Activation.cpp
DEFINE_DISPATCH(relu_stub);
```

**Step 4: CPU 实现**
```cpp
// cpu/Activation.cpp
#include <ATen/native/cpu/Loops.h>

namespace {
Tensor relu_kernel(const Tensor& self) {
  auto iter = at::TensorIteratorConfig()
      .add_output(self)
      .add_input(self)
      .build();
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "relu_cpu", [&]() {
    at::native::cpu_kernel_vec(
        iter,
        [](scalar_t x) { return x > 0 ? x : 0; },
        [](Vectorized<scalar_t> x) { 
          return x > Vectorized<scalar_t>(0); 
        });
  });
  
  return iter.output();
}
} // namespace

REGISTER_DISPATCH(relu_stub, &relu_kernel);
```

---

## 9. 性能优化技巧

### 9.1 使用矢量化

```cpp
// 差：标量版本
cpu_kernel(iter, [](float x) { return x * 2; });

// 好：矢量化版本
cpu_kernel_vec(
    iter,
    [](float x) { return x * 2; },
    [](Vectorized<float> x) { return x * Vectorized<float>(2); });
```

### 9.2 避免不必要的类型转换

```cpp
// 差：在循环内转换
AT_DISPATCH_ALL_TYPES(dtype, "op", [&]() {
  auto factor = double_value;  // 每次迭代都转换
});

// 好：在循环外转换
auto factor = double_value.to<scalar_t>();
AT_DISPATCH_ALL_TYPES(dtype, "op", [&]() {
  // 使用 factor
});
```

### 9.3 使用 TensorIterator 的内置优化

```cpp
// TensorIterator 自动处理：
// 1. 广播
// 2. 类型提升
// 3. 并行化
// 4. 内存布局优化 (channels_last 等)

// 让 TensorIterator 工作
iter.for_each([=](char** data, int64_t n) {
  // 简单的逐元素循环
});
```

---

## 10. 常见错误与排查

### 10.1 错误：Unsupported dtype

```
RuntimeError: "add_cpu" not implemented for 'Bool'

原因：使用了 AT_DISPATCH_ALL_TYPES，但 bool 不在支持列表中
解决：使用 AT_DISPATCH_ALL_TYPES_AND_BOOL
```

### 10.2 错误：ZeroDivisionError

```
RuntimeError: ZeroDivisionError

原因：整数除法未检查除零
解决：添加 TORCH_CHECK(b != 0, "ZeroDivisionError")
```

### 10.3 性能问题

```bash
# 检查是否使用了矢量化
TORCH_SHOW_CPP_STACKTRACES=1 python script.py

# 查看实际调用的 Kernel
gdb -p <pid>
(gdb) break relu_kernel
(gdb) continue
```

---

## 11. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `DispatchStub.h` | L10-L50 | DispatchStub 使用说明 |
| `DispatchStub.h` | L61-L74 | CPUCapability 枚举 |
| `DispatchStub.h` | L96-L214 | DispatchStubImpl 结构 |
| `DispatchStub.h` | L468-L492 | REGISTER_DISPATCH 宏 |
| `TensorIterator.h` | L32-L66 | TensorIterator 配置示例 |
| `TensorIterator.h` | L117-L200 | OperandInfo 结构 |
| `Dispatch.h` | L191-L196 | AT_DISPATCH_FLOATING_TYPES |
| `BinaryOpsKernel.cpp` | L42-L67 | add_clamp_cpu 实现 |
| `BinaryOpsKernel.cpp` | L69-L85 | atan2_kernel 实现 |
| `BinaryOpsKernel.cpp` | L413-L422 | bitwise_and_kernel 实现 |

---

## 12. 下一步

| Part | 主题 |
|------|------|
| [Part 5](./05-registration.md) | 注册机制详解 |
| [Part 6](./06-dispatch-macro.md) | AT_DISPATCH 宏详解 |

---

**参考资料**:
- `aten/src/ATen/native/DispatchStub.h` - DispatchStub 完整定义
- `aten/src/ATen/native/TensorIterator.h` - TensorIterator 配置
- `aten/src/ATen/native/cpu/Loops.h` - CPU Kernel 辅助函数
- `aten/src/ATen/Dispatch.h` - AT_DISPATCH 宏家族
