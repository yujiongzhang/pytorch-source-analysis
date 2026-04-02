# ATen - 核心算子库：文档规划

## 一、文档目标

本系列文档旨在帮助新手理解 PyTorch 的核心算子库 ATen (All Tensor library)。ATen 是 PyTorch 的 C++ 核心，所有 Tensor 操作最终都会落到 ATen 的实现。

**目标读者**:
- 有 PyTorch 使用经验的开发者
- 想了解 PyTorch 底层实现的工程师
- 希望扩展 PyTorch 算子的贡献者

**阅读前提**:
- 熟悉 Python 和 C++ 基础
- 了解 PyTorch 基本 API (Tensor, autograd 等)
- 无需 C++/CUDA 编程经验 (但有帮助)

---

## 二、整体结构规划

### Part 1: ATen 简介与架构

**内容概要**:
- ATen 是什么，为什么需要 C++ 核心
- ATen 在 PyTorch 整体架构中的位置
- 核心文件目录结构
- 从 Python 调用到 C++ 实现的完整链路

**关键源码**:
- `aten/src/ATen/` - ATen 主目录
- `aten/src/ATen/Tensor.h` - Tensor C++ 类定义
- `torch/csrc/autograd/` - Python 绑定层

---

### Part 2: 算子声明系统 - native_functions.yaml

**内容概要**:
- 算子声明的 YAML 格式
- 函数签名语法规则 (参数类型、返回值、默认值)
- Overload 机制
- 变体生成 (function/method/out 版本)
- 别名 (Alias) 机制

**关键源码**:
- `aten/src/ATen/native/native_functions.yaml` - 所有算子声明 (L1-L500+)
- `torchgen/` - 代码生成模块

**示例算子**:
- `abs` (L340-L366) - 基础算子声明
- `native_dropout` (L286-L302) - 多后端 dispatch

---

### Part 3: Dispatch 机制 - 核心中的核心

**内容概要**:
- Dispatcher 单例模式与注册表
- DispatchKey 体系 (CPU, CUDA, Autograd, 等)
- 算子查找与内核选择流程
- BackendFallback 机制
- CompositeImplicitAutograd 与 CompositeExplicitAutograd

**关键源码**:
- `aten/src/ATen/core/dispatch/Dispatcher.h` (L71-L436)
- `aten/src/ATen/core/dispatch/Dispatcher.cpp`
- DispatchKey 枚举定义

**核心概念**:
- OperatorHandle - 算子句柄
- KernelFunction - 内核函数包装
- DispatchTable - 分发查找表

---

### Part 4: 算子实现 - C++ Kernel 编写

**内容概要**:
- C++ 实现文件组织 (按功能模块划分)
- DispatchStub 机制 (CPU 指令集调度)
- 常见 Kernel 模式:
  - 逐元素运算 (Pointwise)
  - 归约运算 (Reduction)
  - 矩阵运算 (BLAS/LAPACK)
- TensorIterator 工具类

**关键源码**:
- `aten/src/ATen/native/DispatchStub.h` - DispatchStub 声明
- `aten/src/ATen/native/activation.cpp` - 激活函数实现
- `aten/src/ATen/native/BinaryOps.cpp` - 二元运算
- `aten/src/ATen/native/cpu/*.cpp` - CPU 优化实现
- `aten/src/ATen/native/cuda/*.cu` - CUDA 实现

**示例分析**:
- ReLU 实现
- Add/Mul 实现
- Softmax 实现

---

### Part 5: 注册机制 - 从声明到实现

**内容概要**:
- REGISTER_DISPATCH 宏
- 算子自动注册流程
- 手动注册 vs 自动生成
- 自定义算子注册

**关键源码**:
- `aten/src/ATen/native/DispatchStub.h` - 注册宏定义
- 各 cpp/cu 文件中的 REGISTER_DISPATCH 调用

**代码模板**:
```cpp
// 1. 声明 stub
using fn_type = void(*)(...);
DECLARE_DISPATCH(fn_type, stub);

// 2. 定义 stub
DEFINE_DISPATCH(stub);

// 3. 注册实现
REGISTER_DISPATCH(stub, &kernel_impl);
```

---

### Part 6: AT_DISPATCH 宏与类型分发

**内容概要**:
- 为什么需要类型分发
- AT_DISPATCH_* 宏家族
- ScalarType 枚举
- 量化类型支持
- 新版 AT_DISPATCH_V2

**关键源码**:
- `aten/src/ATen/Dispatch.h` - 分发宏定义 (L96-L200+)
- `aten/src/ATen/Dispatch_v2.h` - 新版分发宏

**使用模式**:
```cpp
AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "op_name", [&] {
    // scalar_t 已定义，可直接使用
});
```

---

### Part 7: Tensor 核心数据结构

**内容概要**:
- Tensor 与 TensorImpl 分离设计
- Storage 与数据 ownership
- 元数据管理 (size, stride, dtype, device)
- 视图 (View) vs 张量拷贝
- 内存布局 (contiguous, channels_last)

**关键源码**:
- `aten/src/ATen/core/Tensor.h`
- `aten/src/ATen/core/TensorBody.h`
- `aten/src/ATen/Storage.h`

---

### Part 8: 自动微分集成

**内容概要**:
- Autograd 与 ATen 的边界
- derivatives.yaml 配置
- 前向/后向公式注册
- CompositeImplicitAutograd 自动微分
- 自定义算子的 autograd 支持

**关键源码**:
- `tools/autograd/derivatives.yaml`
- `aten/src/ATen/native/AutogradComposite.cpp`

---

### Part 9: 后端扩展指南

**内容概要**:
- 如何添加新后端 (MPS/XLA 等案例)
- BackendFallback 实现
- 必须实现的算子集合
- 测试与验证

**关键源码**:
- `aten/src/ATen/native/mps/` - MPS 后端参考
- `aten/src/ATen/native/cuda/` - CUDA 后端参考

---

### Part 10: 调试与测试

**内容概要**:
- ATen 测试框架
- 算子正确性测试
- 性能测试方法
- 常见错误与排查

**关键源码**:
- `test/` - 测试目录
- `torch/testing/` - 测试工具

---

## 三、源码阅读方法论

### 3.1 由上至下阅读法

```
Python 调用 → Python 绑定 → Dispatcher → Kernel 实现
```

从熟悉的 Python API 出发，追踪到 C++ 实现。

### 3.2 算子追踪法

选择一个简单算子 (如 `abs`, `relu`)，完整追踪:
1. Python 调用
2. 绑定的 C++ 函数
3. Dispatcher 查找
4. 具体 Kernel 实现
5. 底层计算

### 3.3 关键工具

```bash
# 定位算子声明
grep -n "^- func: abs" aten/src/ATen/native/native_functions.yaml

# 定位实现函数
rg "abs_out\(" aten/src/ATen/native/

# 查找 REGISTER_DISPATCH
rg "REGISTER_DISPATCH.*abs" aten/

# 查看调用链
rg --type py "def abs" torch/
```

---

## 四、学习路线建议

### 入门 (Part 1-3)
1. 了解 ATen 整体架构
2. 理解 YAML 声明格式
3. 掌握 Dispatch 机制原理

### 进阶 (Part 4-6)
4. 能读懂 C++ Kernel 实现
5. 理解注册流程
6. 掌握类型分发技巧

### 深入 (Part 7-10)
7. 理解 Tensor 内部结构
8. 了解 autograd 集成
9. 能够扩展新算子/新后端
10. 能够调试和测试

---

## 五、术语表

| 术语 | 说明 |
|------|------|
| ATen | All Tensor library, PyTorch 的 C++ 核心 |
| Dispatcher | 算子分发系统，根据 DispatchKey 选择内核 |
| DispatchKey | 标识后端类型的枚举 (CPU/CUDA/Autograd 等) |
| Kernel | 具体实现某个算子的函数 |
| DispatchStub | CPU 指令集特定的分发机制 |
| TensorIterator | 迭代 Tensor 元素的工具类 |
| CompositeImplicitAutograd | 自动支持 autograd 的复合算子 |

---

## 六、附录：关键文件索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `native_functions.yaml` | L1-L50 | 算子声明格式说明 |
| `native_functions.yaml` | L274-L385 | dispatch 关键字说明 |
| `Dispatcher.h` | L71-L132 | Dispatcher 单例定义 |
| `Dispatcher.h` | L178-L201 | 调用接口 |
| `DispatchStub.h` | L10-L50 | DispatchStub 说明 |
| `Dispatch.h` | L96-L177 | AT_DISPATCH 宏说明 |
| `Tensor.h` | L1-L98 | Tensor 类核心方法 |

---

**后续**: 根据本规划，将逐一编写各 Part 的详细分析文档。
