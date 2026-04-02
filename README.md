# PyTorch 源码阅读与分析

本项目是 PyTorch 源码学习与分析文档，基于 PyTorch 官方 v2.11.0 分支。

## 项目目标

- 深入理解 PyTorch 核心模块的实现原理
- 为开发者提供系统的源码阅读指南
- 记录学习过程中的关键发现与心得

## 源码位置

- **PyTorch 源码**: `/home/zhangyujiong/zyj_ws/torch/pytorch`
- **PyTorch 版本**: v2.11.0 (main 分支)

---

## 已完成内容

### 🔥 Inductor 编译器系列（10 Part）

Inductor 是 PyTorch 2.0 引入的新一代深度学习编译器，作为 `torch.compile` 的默认后端。

| 序号 | 文档 | 内容概述 |
|------|------|----------|
| 01 | [整体架构](./_inductor/01-inductor-architecture.md) | 编译流程、核心组件、数据流 |
| 02 | [IR 系统设计](./_inductor/02-inductor-ir-design.md) | TensorBox/StorageBox/Buffer 体系 |
| 03 | [Lowering 机制](./_inductor/03-lowering-mechanism.md) | 算子到 IR 的转换规则 |
| 04 | [FX Passes](./_inductor/04-fx-passes.md) | Pre/Post-Grad 图优化 |
| 05 | [调度算法](./_inductor/05-scheduler.md) | 拓扑排序与算子融合 |
| 06 | [代码生成](./_inductor/06-codegen.md) | Triton/C++ Kernel 生成 |
| 07 | [AOTInductor](./_inductor/07-aotinductor.md) | 离线编译与打包 |
| 08 | [性能优化](./_inductor/08-performance.md) | 内存重排序与 Combo Kernel |
| 09 | [max-autotune](./_inductor/09-max-autotune.md) | 自动调优系统 |
| 10 | [调试实战](./_inductor/10-debugging.md) | 调试工具与技巧 |

**补充文档**:
- [AOTI 运行时分析](./_inductor/aoti_runtime_analysis.md)
- [博客系列大纲](./_inductor/inductor_blog_series.md)
- [文档修正说明](./_inductor/CORRECTIONS.md)

---

## 其他学习内容（规划中）

以下模块按推荐学习顺序排列：

### 1. TorchDynamo - 字节码捕获

- **源码位置**: `torch/_dynamo/`
- **核心功能**: 通过 Python 字节码分析捕获计算图
- **关键文件**:
  - `eval_frame.py` - 评估帧替换
  - `convert_frame.py` - 帧转换逻辑
  - `bytecode_analysis.py` - 字节码分析
- **学习重点**: 字节码解释、图捕获机制、Guard 系统

### 2. AOTAutograd - 自动微分

- **源码位置**: `torch/_functorch/`
- **核心功能**: 提前模式（Ahead-of-Time）自动微分
- **关键文件**:
  - `aot_autograd.py` - AOT 自动微分主逻辑
  - `functionalize.py` - 函数化转换
- **学习重点**: 前向/后向分离、函数化、View 处理

### 3. FX - 函数式中间表示

- **源码位置**: `torch/fx/`
- **核心功能**: 图表示与转换基础设施
- **关键文件**:
  - `graph.py` - 图数据结构
  - `interpreter.py` - 图解释执行
  - `proxy.py` - 代理张量
- **学习重点**: 图构建、模式匹配、图重写

### 4. ATen - 核心算子库 ✅

- **源码位置**: `aten/src/ATen/`
- **核心功能**: 底层张量操作实现
- **关键文件**:
  - `src/ATen/` - C++ 算子实现
  - `native/` - 原生实现后端
  - `native/native_functions.yaml` - 算子声明
  - `core/dispatch/` - Dispatch 机制
- **学习重点**: 算子注册、Dispatch 机制、Kernel 实现

| 序号 | 文档 | 内容概述 |
|------|------|----------|
| 00 | [文档规划](./_aten/00-plan.md) | 整体结构、学习路线 |
| 01 | [架构概览](./_aten/01-architecture.md) | ATen 是什么、目录结构、核心概念 |
| 02 | [算子声明系统](./_aten/02-native-functions-yaml.md) | native_functions.yaml 格式详解 |
| 03 | [Dispatch 机制](./_aten/03-dispatch-mechanism.md) | Dispatcher、DispatchKey、注册与调用 |
| 04 | [C++ Kernel 实现](./_aten/04-kernel-implementation.md) | DispatchStub、TensorIterator、Kernel 模式 |
| 05 | [注册机制](./_aten/05-registration.md) | REGISTER_DISPATCH 宏、算子注册流程 |
| 06 | [AT_DISPATCH 宏](./_aten/06-dispatch-macro.md) | 类型分发宏家族、AT_DISPATCH_V2 |
| 07 | [Tensor 数据结构](./_aten/07-tensor-structure.md) | TensorImpl、Storage、视图机制 |
| 08 | [自动微分集成](./_aten/08-autograd-integration.md) | derivatives.yaml、AutogradComposite |
| 09 | [后端扩展指南](./_aten/09-backend-extension.md) | 新后端实现、BackendFallback |
| 10 | [调试与测试](./_aten/10-debugging-testing.md) | 测试框架、调试工具、OpInfo |

### 5. 分布式训练

- **源码位置**: `torch/distributed/`
- **核心功能**: 多机多卡训练支持
- **关键模块**:
  - `distributed/` - 核心分布式原语
  - `_spmd/` - SPMD 编程模型
  - `_composable/` - 可组合并行策略
- **学习重点**: DDP、FSDP、流水线并行、张量并行

### 6. CUDA 后端支持

- **源码位置**: `torch/cuda/`, `torch/csrc/cuda/`
- **核心功能**: NVIDIA GPU 后端支持
- **关键文件**:
  - `cuda/__init__.py` - CUDA 初始化
  - `csrc/cuda/` - CUDA C++ 绑定
- **学习重点**: Stream 管理、内存分配、Kernel 启动

### 7. 量化（Quantization）

- **源码位置**: `torch/quantization/`
- **核心功能**: 模型量化压缩
- **关键模块**:
  - `quantization/` - 量化 API
  - `ao/` - 高级优化
- **学习重点**: PTQ、QAT、量化算子、校准

### 8. Python 前端与 Tensor 系统

- **源码位置**: `torch/_tensor.py`, `torch/_torch_docs.py`
- **核心功能**: Python 层 Tensor API
- **关键文件**:
  - `_tensor.py` - Tensor 类定义
  - `_torch_docs.py` - 文档字符串
  - `storage.py` - 存储管理
- **学习重点**: Tensor 生命周期、Storage、Dispatcher

### 9. 内存管理

- **源码位置**: `torch/csrc/allocators/`
- **核心功能**: 内存分配与回收
- **关键模块**:
  - CUDA 缓存分配器
  - 异步分配器
- **学习重点**: 缓存池、碎片整理、分配策略

### 10. 测试框架

- **源码位置**: `torch/testing/`
- **核心功能**: 内部测试基础设施
- **关键文件**:
  - `_internal/common_utils.py` - 通用测试工具
- **学习重点**: 测试用例编写、断言工具、CI 集成

---

## 学习方法建议

### 1. 由浅入深

```
FX Graph → Dynamo → AOTAutograd → Inductor → Codegen
```

建议从 FX 开始，理解图表示，再学习编译流程。

### 2. 结合调试工具

```bash
# 启用 Inductor 调试输出
TORCH_COMPILE_DEBUG=1 python your_script.py

# 查看生成的 Triton Kernel
TORCH_LOGS="output_code" python your_script.py

# 查看编译过程
TORCH_LOGS="graph_breaks,recompiles" python your_script.py
```

### 3. 使用源码定位工具

```bash
# 定位类定义
grep -rn "class GraphLowering" torch/_inductor/

# 定位函数定义
rg "def compile_fx_inner"

# 查看调用关系
rg --type py "compile_fx_inner" -A 3
```

### 4. 阅读测试用例

PyTorch 测试用例是很好的学习资源：

```bash
# Inductor 测试
python test/inductor/test_torchinductor.py

# Dynamo 测试
python test/dynamo/test_dynamo.py
```

---

## 参考资料

### 官方资源

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [PyTorch 2.0 Release Notes](https://github.com/pytorch/pytorch/releases)

### 技术博客

- [PyTorch Compiler Internals](https://dev-discuss.pytorch.org/)
- [Triton 编程语言](https://triton-lang.org/)

---

## 贡献指南

欢迎贡献源码分析文档！提交前请确保：

1. 代码示例经过验证
2. 引用源码行号准确（使用 `grep` 验证）
3. 架构图清晰易懂
4. 术语使用准确

---

## 许可证

本项目遵循 PyTorch 开源许可证。
