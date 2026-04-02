# PyTorch AOT Inductor Runtime 深度解析：CUDA 后端 .so 文件加载与执行全流程

## 前言

PyTorch 的 AOT Inductor (Ahead-of-Time Inductor，简称 AOTI) 是 PyTorch 2.x 引入的重要编译技术，允许用户将训练好的模型预编译成独立的共享库文件 (.so)，实现离线推理部署。本文将深入分析 AOTI runtime 的完整工作流程，特别是 CUDA 后端如何加载和执行编译后的 .so 文件。

**示例代码：**

```python
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "model.pt2"))
print(model(torch.randn(8, 10, device=device)))
```

这看似简单的几行代码背后，隐藏着复杂的加载和执行机制。让我们一起揭开这层神秘的面纱。

## 一、整体架构概览

### 1.1 AOTI 编译与运行时架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AOTI 完整工作流程                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   编译期 (Compile Time)                    运行期 (Run Time)                 │
│   ──────────────────                       ──────────────────               │
│                                                                             │
│   ┌──────────────┐                                                      │
│   │ Exported     │                                                      │
│   │ Program      │                                                      │
│   │ (FX Graph)   │                                                      │
│   └──────┬───────┘                                                      │
│          │ aoti_compile_and_package()                                    │
│          ▼                                                               │
│   ┌──────────────┐         ┌──────────────┐                             │
│   │ Codegen      │────────▶│  .cpp 文件    │  用户态加载                     │
│   │ (Triton/C++) │         │  + .o 文件    │  ─────────────────────────▶  │
│   └──────────────┘         └──────┬───────┘                             │
│                                   │                                     │
│                                   ▼                                     │
│                            ┌──────────────┐                             │
│                            │   .pt2       │  (ZIP 格式 Archive)            │
│                            │   Package    │                             │
│                            └──────┬───────┘                             │
│                                   │                                     │
│                                   ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │                    C++ Runtime Layer                            │    │
│   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │    │
│   │  │ AOTIModel       │  │ AOTIModel       │  │ Model Container │ │    │
│   │  │ PackageLoader   │──│ ContainerRunner │──│ & Model         │ │    │
│   │  │ (加载器)         │  │ (执行器)         │  │ (模型实例)       │ │    │
│   │  └─────────────────┘  └─────────────────┘  └─────────────────┘ │    │
│   └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件分层

AOTI runtime 分为以下几个关键层次：

| 层级 | 组件 | 文件位置 | 功能描述 |
|------|------|----------|----------|
| Python API 层 | `aoti_load_package` | `torch/_inductor/__init__.py` | 用户入口，调用 package 加载 |
| Package 层 | `load_package` | `torch/_inductor/package/package.py` | 解析 .pt2 archive，创建 C++ loader |
| C++ Binding 层 | `AOTIModelPackageLoader` | `torch/csrc/inductor/aoti_package/` | 解压 archive，编译/加载 .so |
| Runner 层 | `AOTIModelContainerRunner` | `torch/csrc/inductor/aoti_runner/` | 设备相关的模型执行器 |
| CUDA Runner | `AOTIModelContainerRunnerCuda` | `torch/csrc/inductor/aoti_runner/model_container_runner_cuda.*` | CUDA 后端具体实现 |
| Runtime Interface | `AOTInductorModelContainer*` | `torch/csrc/inductor/aoti_runtime/` | .so 中导出的 C API |
| Model 层 | `AOTInductorModel` | `torch/csrc/inductor/aoti_runtime/` | 单个模型实例，持有 kernels |

---

## 二、Python 层入口分析

### 2.1 `aoti_load_package` 入口

**源码位置：** `torch/_inductor/__init__.py:239-269`

```python
def aoti_load_package(
    path: FileLike, run_single_threaded: bool = False, device_index: int = -1
) -> AOTICompiledModel:
    """
    Loads the model from the PT2 package.
    """
    from torch._inductor.package import load_package

    return load_package(
        path, run_single_threaded=run_single_threaded, device_index=device_index
    )
```

这是用户调用的入口函数，它委托给 `torch._inductor.package.load_package`。

### 2.2 `load_package` 实现

**源码位置：** `torch/_inductor/package/package.py:102-138`

```python
def load_package(
    path: FileLike,
    model_name: str = "model",
    run_single_threaded: bool = False,
    num_runners: int = 1,
    device_index: int = -1,
) -> AOTICompiledModel:
    try:
        pt2_contents = load_pt2(
            path,
            run_single_threaded=run_single_threaded,
            num_runners=num_runners,
            device_index=device_index,
        )
        if model_name not in pt2_contents.aoti_runners:
            raise RuntimeError(f"Model {model_name} not found in package")
        return pt2_contents.aoti_runners[model_name]
    except RuntimeError:
        log.warning("Loading outdated pt2 file. Please regenerate your package.")

    # Fallback: 直接使用 C++ loader
    path = os.fspath(path)
    loader = torch._C._aoti.AOTIModelPackageLoader(
        path, model_name, run_single_threaded, num_runners, device_index
    )
    return AOTICompiledModel(loader)
```

核心逻辑：
1. 首先尝试通过 `load_pt2` 加载（新版格式，支持 weights 分离）
2. 如果失败，回退到直接使用 `torch._C._aoti.AOTIModelPackageLoader`

### 2.3 `AOTICompiledModel` 包装类

**源码位置：** `torch/export/pt2_archive/_package.py:719-738`

```python
class AOTICompiledModel:
    """
    Callable AOT Inductor loaded model from a .pt2
    """

    def __init__(self, loader: torch._C._aoti.AOTIModelPackageLoader) -> None:
        self.loader = loader

    def __call__(self, *args, **kwargs):
        call_spec = self.loader.get_call_spec()
        in_spec = pytree.treespec_loads(call_spec[0])
        out_spec = pytree.treespec_loads(call_spec[1])
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        flat_inputs = [x for x in flat_inputs if isinstance(torch.Tensor)]
        flat_outputs = self.loader.boxed_run(flat_inputs)
        return pytree.tree_unflatten(flat_outputs, out_spec)
```

这个类包装了 C++ loader，实现了：
- 输入/输出 pytree spec 的解析
- 张量 flattening/unflattening
- 调用 `boxed_run` 执行实际推理

---

## 三、C++ Package Loader 层

### 3.1 `AOTIModelPackageLoader` 构造函数

**源码位置：** `torch/csrc/inductor/aoti_package/model_package_loader.cpp:663-828`

这是整个加载流程的核心，让我们详细分析：

```cpp
AOTIModelPackageLoader::AOTIModelPackageLoader(
    const std::string& model_package_path,
    const std::string& model_name,
    const bool run_single_threaded,
    const size_t num_runners,
    const c10::DeviceIndex device_index) {
  
  // ========== 步骤 1: 解压 .pt2 archive (ZIP 格式) ==========
  RAIIMinizArchive zip_archive{model_package_path};
  auto found_filenames{zip_archive.get_filenames()};
  
  // 创建临时目录存放解压的文件
  temp_dir_ = normalize_path_separator(create_temp_dir());
  
  std::string so_filename;
  std::string cpp_filename;
  std::vector<std::string> obj_filenames;
  std::string model_directory = ...; // 构建模型目录路径
  
  // 解压所有相关文件到临时目录
  for (auto const& zip_filename_str : found_filenames) {
    if (c10::starts_with(cur_filename, model_directory) ||
        c10::starts_with(cur_filename, const_directory)) {
      zip_archive.extract_file(zip_filename_str, output_path_str);
      
      // 记录 .so/.cpp/.o/.blob 文件路径
      if (filename_extension == ".cpp") {
        cpp_filename = output_file_path;
      } else if (filename_extension == extension_file_ext()) {
        so_filename = output_file_path;
      } else if (filename_extension == ".blob") {
        weight_blob_filename = output_file_path;
      }
    }
  }
  
  // ========== 步骤 2: 编译 .so (如果没有预编译的 .so) ==========
  std::string so_path = !so_filename.empty()
      ? so_filename
      : compile_so(cpp_filename, obj_filenames);
  
  // ========== 步骤 3: 加载 metadata ==========
  load_metadata(cpp_filename);
  
  // ========== 步骤 4: 根据 device 类型创建 runner ==========
  std::string device_key = metadata_["AOTI_DEVICE_KEY"];
  TORCH_CHECK(!device_key.empty(), "No device information found.");
  
  // 从 registry 获取对应设备的 runner 创建函数
  std::unordered_map<std::string, CreateAOTIModelRunnerFunc>
      registered_aoti_runner = getAOTIModelRunnerRegistry();
  
  TORCH_CHECK(
      registered_aoti_runner.find(device_key) != registered_aoti_runner.end(),
      "Unsupported device key found: ",
      device_key);
  
  c10::Device device = c10::Device(device_key);
  device.set_index(device_index);
  
  std::string cubin_dir = temp_dir_ + k_separator + model_directory;
  
  // 创建 runner 实例
  runner_ = registered_aoti_runner[device_key](
      so_path, num_runners, device.str(), cubin_dir, run_single_threaded);
  
  // ========== 步骤 5: 加载 weights (如果有 .blob 文件) ==========
  if (!weight_blob_filename.empty()) {
    runner_->update_constant_buffer_from_blob(weight_blob_filename);
  }
}
```

### 3.2 .so 编译流程

如果 .pt2 包中只有 .cpp 文件而没有预编译的 .so，则需要在运行时编译：

**源码位置：** `torch/csrc/inductor/aoti_package/model_package_loader.cpp:388-443`

```cpp
std::string compile_so(
    const std::string& cpp_filename,
    std::vector<std::string>& obj_filenames) {
  
  size_t lastindex = cpp_filename.find_last_of('.');
  std::string filename = cpp_filename.substr(0, lastindex);
  
  // 读取编译 flags
  std::string compile_flags_path = filename + "_compile_flags.json";
  const nlohmann::json compile_flags = load_json_file(compile_flags_path);
  
  auto [compile_cmd, output_o] =
      get_cpp_compile_command(filename, {cpp_filename}, compile_flags);
  
  // 读取链接 flags
  std::string linker_flags_path = filename + "_linker_flags.json";
  const nlohmann::json linker_flags = load_json_file(linker_flags_path);
  
  obj_filenames.push_back(output_o);
  auto [link_cmd, output_so] =
      get_cpp_compile_command(filename, obj_filenames, linker_flags);
  
  // 执行编译和链接命令
  TORCH_CHECK(system(compile_cmd.c_str()) == 0, "Failed to compile cpp file.");
  TORCH_CHECK(system(link_cmd.c_str()) == 0, "Failed to link files.");
  
  // 将 mmap weights 追加到 .so 文件末尾
  if (file_exists(serialized_weights_path)) {
    // ... 将 weights 以 page-aligned 方式追加到 .so
  }
  
  return output_so;
}
```

---

## 四、Model Container Runner 层

### 4.1 Runner Registry 机制

AOTI 使用注册表模式来支持多后端（CUDA、CPU、MPS、XPU 等）：

**源码位置：** `torch/csrc/inductor/aoti_runner/model_container_runner.cpp:380-385`

```cpp
std::unordered_map<std::string, CreateAOTIModelRunnerFunc>&
getAOTIModelRunnerRegistry() {
  static std::unordered_map<std::string, CreateAOTIModelRunnerFunc>
      aoti_model_runner_registry_;
  return aoti_model_runner_registry_;
}
```

### 4.2 CUDA Runner 注册

**源码位置：** `torch/csrc/inductor/aoti_runner/model_container_runner_cuda.cpp:38-52`

```cpp
namespace {
std::unique_ptr<AOTIModelContainerRunner> create_aoti_runner_cuda(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir,
    const bool run_single_threaded) {
  return std::make_unique<AOTIModelContainerRunnerCuda>(
      model_so_path, num_models, device_str, cubin_dir, run_single_threaded);
}
} // namespace

// 静态注册：在程序启动时自动注册到 registry
RegisterAOTIModelRunner register_cuda_runner("cuda", &create_aoti_runner_cuda);
```

当 `AOTIModelPackageLoader` 查询 `device_key = "cuda"` 时，就会调用这个注册函数创建 `AOTIModelContainerRunnerCuda` 实例。

### 4.3 Runner 基类：加载 .so 并解析符号

**源码位置：** `torch/csrc/inductor/aoti_runner/model_container_runner.cpp:24-126`

```cpp
AOTIModelContainerRunner::AOTIModelContainerRunner(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir,
    const bool run_single_threaded) {
  
  // ========== 步骤 1: 使用 DynamicLibrary 加载 .so ==========
  model_so_ = std::make_unique<at::DynamicLibrary>(model_so_path.c_str());
  TORCH_CHECK(model_so_, "Failed to load model: ", model_so_path);
  
  // ========== 步骤 2: 通过 dlsym 加载所有必要的函数符号 ==========
#define LOAD_SYMBOL(var, name_str) \
  var = reinterpret_cast<decltype(var)>(model_so_->sym(name_str));
  
  // 核心函数符号
  LOAD_SYMBOL(create_func_, "AOTInductorModelContainerCreateWithDevice")
  LOAD_SYMBOL(delete_func_, "AOTInductorModelContainerDelete")
  LOAD_SYMBOL(get_num_outputs_func_, "AOTInductorModelContainerGetNumOutputs")
  LOAD_SYMBOL(get_num_constants_func_, "AOTInductorModelContainerGetNumConstants")
  LOAD_SYMBOL(get_constant_name_func_, "AOTInductorModelContainerGetConstantName")
  LOAD_SYMBOL(get_constant_original_fqn_func_, "AOTInductorModelContainerGetConstantOriginalFQN")
  LOAD_SYMBOL(get_constant_dtype_func_, "AOTInductorModelContainerGetConstantDtype")
  LOAD_SYMBOL(update_constant_buffer_func_, "AOTInductorModelContainerUpdateConstantBuffer")
  LOAD_SYMBOL(run_const_fold_func_, "AOTInductorModelContainerRunConstantFolding")
  LOAD_SYMBOL(swap_constant_buffer_func_, "AOTInductorModelContainerSwapConstantBuffer")
  LOAD_SYMBOL(get_call_spec_func_, "AOTInductorModelContainerGetCallSpec")
  
  // 可选函数符号（根据模型版本可能不存在）
  TRY_LOAD_SYMBOL(run_func_, run_func_name)
  TRY_LOAD_SYMBOL(free_inactive_constant_buffer_func_, ...)
  TRY_LOAD_SYMBOL(extract_constants_map_func_, ...)
  TRY_LOAD_SYMBOL(update_constants_from_blob_func_, ...)
#undef LOAD_SYMBOL
  
  // ========== 步骤 3: 加载 Proxy Executor (用于 const folding) ==========
  size_t lastindex = model_so_path.find_last_of('.');
  std::string json_filename = model_so_path.substr(0, lastindex) + ".json";
  
  if (c10::filesystem::exists(json_filename)) {
    proxy_executor_ = std::make_unique<torch::aot_inductor::OSSProxyExecutor>(
        json_filename, device_str == "cpu");
    proxy_executor_handle_ =
        reinterpret_cast<AOTIProxyExecutorHandle>(proxy_executor_.get());
  } else {
    proxy_executor_handle_ = nullptr;
  }
  
  // ========== 步骤 4: 调用 .so 中的创建函数，创建 container handle ==========
  AOTI_RUNTIME_ERROR_CODE_CHECK(create_func_(
      &container_handle_,
      num_models,
      device_str.c_str(),
      cubin_dir.empty() ? nullptr : cubin_dir.c_str()));
}
```

**关键点：**
1. 使用 `at::DynamicLibrary`（封装 dlopen/dlsym）加载 .so
2. 从 .so 中导出所有必要的 C API 函数指针
3. 调用 `.so` 中的 `AOTInductorModelContainerCreateWithDevice` 创建实际的模型容器

---

## 五、.so 内部的 Runtime 实现

### 5.1 C API 接口定义

**源码位置：** `torch/csrc/inductor/aoti_runtime/interface.h`

这是编译生成的 .so 文件导出的 C API 接口：

```cpp
extern "C" {
// 创建模型容器
AOTI_API AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir);

// 删除模型容器
AOTI_API AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle);

// 执行推理
AOTI_API AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles,
    size_t num_inputs,
    AtenTensorHandle* output_handles,
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

// 单线程版本
AOTI_API AOTIRuntimeError AOTInductorModelContainerRunSingleThreaded(...);

// 常量相关 API
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetNumConstants(...);
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetConstantName(...);
AOTI_API AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(...);
// ... 更多 API
}
```

### 5.2 C API 实现

**源码位置：** `torch/_inductor/codegen/aoti_runtime/interface.cpp`

这个文件在编译时会被嵌入到生成的 .so 文件中：

```cpp
AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir) {

  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0\n";
    return AOTI_RUNTIME_FAILURE;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    // 创建 C++ 模型容器对象
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, std::string(device_str), cubin_dir_opt);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles,
    size_t num_inputs,
    AtenTensorHandle* output_handles,
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");
  
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;  // 推理时禁用 grad
    container->run(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}
```

### 5.3 `AOTInductorModelContainer` 类

**源码位置：** `torch/csrc/inductor/aoti_runtime/model_container.h`

这是 .so 内部的核心类，管理模型实例和并发执行：

```cpp
class AOTInductorModelContainer {
 public:
  AOTInductorModelContainer(
      size_t num_models,
      const std::string& device_str,
      const std::optional<std::string>& cubin_dir = std::nullopt) {
    
    constants_map_ = std::make_shared<ConstantMap>();
    constants_array_ = std::make_shared<std::vector<ConstantHandle>>();
    
    // 创建多个模型实例用于并发执行
    models_.reserve(num_models);
    available_models_.reserve(num_models);
    for (size_t i = 0; i < num_models; ++i) {
      models_.push_back(AOTInductorModel::Create(
          constants_map_, constants_array_, device_str, cubin_dir));
      available_models_.push_back(models_.back().get());
    }
    
    // 从第一个模型加载常量信息
    auto* model = available_models_[0];
    model->load_constants();
    constant_blob_ = model->release_constant_blob();
    
    // 计算常量 blob 的内存布局
    model->compute_constant_blob(
        blob_size_,
        constants_internal_offset_,
        secondary_cpu_blob_size_,
        secondary_cpu_constants_internal_offset_);
    
    constant_folded_ = ConstantState::INITIALIZED;
    
    // 同步所有模型的常量映射
    for (auto& model : models_) {
      model->update_constants_map(constants_map_);
    }
  }
  
  void run(
      AtenTensorHandle* input_handles,
      AtenTensorHandle* output_handles,
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor) {
    
    // 线程安全地获取可用模型
    std::shared_lock model_lk(model_exec_mutex_);
    auto* model = get_available_model();
    
    // 检查常量是否已准备就绪，如未就绪则执行 const folding
    if (const_folded == ConstantState::INITIALIZED) {
      model_lk.unlock();
      std::unique_lock constants_folding_lk(model_exec_mutex_);
      if (const_folded == ConstantState::INITIALIZED) {
        auto folded_const_map = model->run_const_fold(
            stream, proxy_executor, /* initialization = */ true);
        update_constant_buffer(std::move(folded_const_map), ...);
        const_folded = ConstantState::FOLDED;
      }
    }
    
    // 执行模型推理
    try {
      model->run(input_handles, output_handles, stream, proxy_executor);
    } catch (...) {
      std::lock_guard lk(models_mutex_);
      available_models_.push_back(model);  // 放回可用队列
      throw;
    }
    
    // 将模型放入待回收队列
    {
      std::lock_guard lk(models_mutex_);
      pending_models_.push_back(model);
    }
    pending_models_available_.notify_one();
  }
  
 private:
  // 多个模型实例用于并发
  std::vector<std::unique_ptr<AOTInductorModel>> models_;
  std::vector<AOTInductorModel*> available_models_;
  std::deque<AOTInductorModel*> pending_models_;
  
  // 线程同步原语
  std::mutex models_mutex_;
  std::condition_variable pending_models_available_;
  std::shared_mutex model_exec_mutex_;
};
```

---

## 六、CUDA 后端特定实现

### 6.1 `AOTIModelContainerRunnerCuda`

**源码位置：** `torch/csrc/inductor/aoti_runner/model_container_runner_cuda.cpp`

```cpp
class AOTIModelContainerRunnerCuda : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerCuda(
      const std::string& model_so_path,
      size_t num_models,
      const std::string& device_str,
      const std::string& cubin_dir,
      const bool run_single_threaded)
    : AOTIModelContainerRunner(
          model_so_path,
          num_models,
          device_str,
          cubin_dir,
          run_single_threaded) {}
  
  std::vector<at::Tensor> run_impl(
      std::vector<AtenTensorHandle>& input_handles,
      void* stream_handle) override {
    // CUDA 特定：获取当前 CUDA stream
    if (stream_handle == nullptr) {
      at::cuda::CUDAStream cuda_stream = c10::cuda::getCurrentCUDAStream();
      stream_handle = reinterpret_cast<void*>(cuda_stream.stream());
    }
    return AOTIModelContainerRunner::run_impl(input_handles, stream_handle);
  }
};
```

### 6.2 CUDA 常量加载

在 `update_constant_buffer` 中，CUDA 后端的常量数据需要从 CPU 复制到 GPU：

**源码位置：** `torch/csrc/inductor/aoti_runner/model_container_runner.cpp:546-554`

```cpp
#if defined(USE_CUDA)
  AOTI_RUNTIME_CUDA_CHECK(cudaMemcpy(
      internal_constants_ptr,
      user_constant_ptr,
      constant_size,
      cudaMemcpyDefault));
#endif
```

---

## 七、完整执行流程图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            AOTI Runtime 完整执行流程 (CUDA 后端)                  │
└─────────────────────────────────────────────────────────────────────────────────┘

  Python 层                            C++ 层                             .so 内部
  ──────────                           ────                              ────────
  
  ┌──────────────────┐
  │ aoti_load_package│
  │ (用户调用入口)    │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ load_package     │
  │ (package.py)     │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ load_pt2         │
  │ (解析 .pt2       │
  │  ZIP archive)    │
  └────────┬─────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                    torch._C._aoti.AOTIModelPackageLoader                │
  │                            (C++ Binding Layer)                          │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  ┌─────────────────┐                                                   │
  │  │ 1. 解压 .pt2    │  提取 .cpp/.so/.o/.json/.blob 到临时目录          │
  │  └────────┬────────┘                                                   │
  │           │                                                           │
  │           ▼                                                           │
  │  ┌─────────────────┐                                                   │
  │  │ 2. 编译 .so     │  如果没有预编译 .so，执行 g++/nvcc 编译链接       │
  │  └────────┬────────┘                                                   │
  │           │                                                           │
  │           ▼                                                           │
  │  ┌─────────────────┐                                                   │
  │  │ 3. 加载 metadata│  读取 _metadata.json 获取设备信息等               │
  │  └────────┬────────┘                                                   │
  │           │                                                           │
  │           ▼                                                           │
  │  ┌─────────────────┐                                                   │
  │  │ 4. 查询 Registry│  getAOTIModelRunnerRegistry()["cuda"]            │
  │  └────────┬────────┘                                                   │
  │           │                                                           │
  │           ▼                                                           │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │ 5. 创建 AOTIModelContainerRunnerCuda                            │  │
  │  │    ┌─────────────────────────────────────────────────────────┐  │  │
  │  │    │ Runner 基类构造函数 (model_container_runner.cpp)        │  │  │
  │  │    │  a) at::DynamicLibrary::open(.so 文件) ← dlopen          │  │  │
  │  │    │  b) dlsym 加载所有 C API 函数指针：                       │  │  │
  │  │    │     - AOTInductorModelContainerCreateWithDevice         │  │  │
  │  │    │     - AOTInductorModelContainerRun                       │  │  │
  │  │    │     - AOTInductorModelContainerDelete                    │  │  │
  │  │    │     - ... (约 20+ 个函数)                                 │  │  │
  │  │    │  c) 加载 ProxyExecutor (用于 const folding)              │  │  │
  │  │    │  d) 调用 create_func_ → AOTInductorModelContainerCreate  │  │  │
  │  │    └─────────────────────────────────────────────────────────┘  │  │
  │  │                                                                  │  │
  │  │    ┌─────────────────────────────────────────────────────────┐  │  │
  │  │    │ .so 内部：AOTInductorModelContainer 构造函数             │  │  │
  │  │    │  a) 创建 N 个 AOTInductorModel 实例 (num_runners)       │  │  │
  │  │    │  b) 每个 Model 包含：                                    │  │  │
  │  │    │     - constants_map_ (常量张量映射)                     │  │  │
  │  │    │     - constants_array_ (常量数组)                       │  │  │
  │  │    │     - kernels_ (指向编译后的 kernel 函数)                 │  │  │
  │  │    │  c) load_constants() - 从 .so 的数据段加载常量           │  │  │
  │  │    │  d) compute_constant_blob() - 计算常量内存布局          │  │  │
  │  │    └─────────────────────────────────────────────────────────┘  │  │
  │  └─────────────────────────────────────────────────────────────────┘  │
  │                                                                         │
  │  ┌─────────────────┐                                                   │
  │  │ 6. 加载 weights │  如果有 .blob 文件，调用 update_constants_from_blob│
  │  └────────┬────────┘                                                   │
  │           │                                                           │
  └───────────┼───────────────────────────────────────────────────────────┘
              │
              ▼
  ┌──────────────────┐
  │ AOTICompiledModel│
  │ (Python 包装类)   │
  └────────┬─────────┘
           │
           │  用户调用 model(input_tensor)
           │
           ▼
  ┌──────────────────┐
  │ __call__         │
  │  - 解析 in_spec   │
  │  - flatten 输入   │
  │  - 调用 loader    │
  │    .boxed_run()  │
  └────────┬─────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                    AOTIModelPackageLoader::boxed_run                    │
  └─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │              AOTIModelContainerRunner::boxed_run                        │
  │   - 将 at::Tensor 转换为 AtenTensorHandle (C ABI 兼容)                   │
  │   - 调用 run_impl()                                                    │
  └─────────────────────────────────────────────────────────────────────────┘
           │
           ▼ (CUDA 后端)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │         AOTIModelContainerRunnerCuda::run_impl                          │
  │   - 获取当前 CUDA stream (c10::cuda::getCurrentCUDAStream())           │
  │   - 调用基类 run_impl(stream_handle)                                   │
  └─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │              AOTIModelContainerRunner::run_impl                         │
  │   - 调用 .so 中的 run_func_ (AOTInductorModelContainerRun)              │
  └─────────────────────────────────────────────────────────────────────────┘
           │
           ▼ (.so 内部)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │           AOTInductorModelContainerRun (interface.cpp)                  │
  │   - 检查输入/输出数量                                                  │
  │   - AOTINoGradGuard (禁用 grad)                                        │
  │   - container->run(...)                                                │
  └─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │        AOTInductorModelContainer::run (model_container.h)               │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  ┌───────────────────────────────────────────────────────────────────┐ │
  │  │ 1. 获取可用模型 (线程安全)                                         │ │
  │  │    - available_models_.back()                                     │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  │                                                                        │
  │  ┌───────────────────────────────────────────────────────────────────┐ │
  │  │ 2. 检查常量状态 (ConstantState)                                   │ │
  │  │    - INITIALIZED → 执行 const folding                            │ │
  │  │      * model->run_const_fold(stream, proxy_executor)             │ │
  │  │      * 使用 OSSProxyExecutor 执行折叠计算                         │ │
  │  │      * 更新常量 buffer                                            │ │
  │  │    - FOLDED → 直接执行                                           │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  │                                                                        │
  │  ┌───────────────────────────────────────────────────────────────────┐ │
  │  │ 3. 执行模型推理                                                   │ │
  │  │    model->run(input_handles, output_handles, stream, ...)        │ │
  │  │                                                                   │ │
  │  │    ┌───────────────────────────────────────────────────────────┐ │ │
  │  │    │ AOTInductorModel::run → run_impl                          │ │ │
  │  │    │  - 调用 kernels_->run()                                   │ │ │
  │  │    │    (这是编译期生成的 kernel 函数)                           │ │ │
  │  │    │                                                            │ │ │
  │  │    │    // 示例生成的 kernel 代码结构：                          │ │ │
  │  │    │    void run(AtenTensorHandle* inputs,                      │ │ │
  │  │    │               AtenTensorHandle* outputs,                   │ │ │
  │  │    │               DeviceStreamType stream) {                   │ │ │
  │  │    │      // 1. 执行 Triton kernel (通过 CUDA launch)            │ │ │
  │  │    │      //    - 调用 cubin 中的 kernel                        │ │ │
  │  │    │      // 2. 执行 C++ kernel (ATen 操作)                     │ │ │
  │  │    │      //    - aoti_torch_cuda_* 系列函数                    │ │ │
  │  │    │      // 3. 管理中间 buffer                                 │ │ │
  │  │    │    }                                                       │ │ │
  │  │    └───────────────────────────────────────────────────────────┘ │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  │                                                                        │
  │  ┌───────────────────────────────────────────────────────────────────┐ │
  │  │ 4. 模型回收                                                       │ │
  │  │    - 将模型放入 pending_models_                                   │ │
  │  │    - notify pending_models_available_                            │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────────────┘
           │
           │  等待模型执行完成 (如果是同步调用)
           │
           ▼
  ┌──────────────────┐
  │ 返回输出 Tensor   │
  │ (Python 层       │
  │  unflatten)      │
  └──────────────────┘
```

---

## 八、关键数据结构

### 8.1 AtenTensorHandle (C ABI 兼容)

```cpp
// 定义在 torch/csrc/inductor/aoti_torch/c/shim.h
struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;
```

这是为了跨 C/C++ 边界兼容而设计的不透明指针类型，实际指向 `at::Tensor`。

### 8.2 ConstantMap

```cpp
using ConstantMap = std::unordered_map<std::string, AtenTensorHandle>;
```

存储模型常量的映射表，key 是常量名，value 是 Tensor handle。

### 8.3 DeviceStreamType

```cpp
// CUDA 后端：指向 CUDA stream
// CPU 后端：nullptr
using DeviceStreamType = void*;
```

---

## 九、总结与要点

### 9.1 核心流程总结

1. **加载阶段**：
   - Python 层调用 `aoti_load_package`
   - C++ `AOTIModelPackageLoader` 解压 .pt2 (ZIP 格式)
   - 编译或加载 .so 文件
   - 根据设备类型从 registry 获取对应的 runner

2. **初始化阶段**：
   - Runner 通过 dlopen/dlsym 加载 .so 中的 C API
   - 调用 .so 中的 `AOTInductorModelContainerCreateWithDevice`
   - 创建多个 `AOTInductorModel` 实例用于并发
   - 加载常量数据（从 .so 数据段或 .blob 文件）

3. **执行阶段**：
   - Python `__call__` → C++ `boxed_run` → `.so` 中的 `run`
   - 线程安全地获取可用模型实例
   - 如需 const folding，先执行常量折叠
   - 调用编译后的 kernel 函数（Triton/C++）
   - CUDA 后端使用 CUDA stream 执行

### 9.2 设计亮点

| 特性 | 实现方式 |
|------|----------|
| **多后端支持** | Runner Registry 模式，静态注册 |
| **并发执行** | 多模型实例 + 线程池管理 |
| **常量折叠** | ProxyExecutor + 惰性求值 |
| **跨平台** | C ABI 接口 + at::DynamicLibrary |
| **Weights 分离** | .blob 文件 + mmap 加载 |

### 9.3 CUDA 后端特定优化

- 使用 CUDA stream 进行异步执行
- 常量数据从 CPU 到 GPU 的 cudaMemcpy
- Cubin 文件加载（如果有预编译的 CUDA kernel）

---

## 十、参考源码文件清单

| 文件路径 | 作用 |
|---------|------|
| `torch/_inductor/__init__.py` | Python API 入口 |
| `torch/_inductor/package/package.py` | Package 加载逻辑 |
| `torch/export/pt2_archive/_package.py` | PT2 archive 解析 |
| `torch/csrc/inductor/aoti_package/model_package_loader.h/cpp` | C++ Package Loader |
| `torch/csrc/inductor/aoti_runner/model_container_runner.h/cpp` | Runner 基类 |
| `torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h/cpp` | CUDA Runner |
| `torch/csrc/inductor/aoti_runtime/interface.h` | C API 接口定义 |
| `torch/_inductor/codegen/aoti_runtime/interface.cpp` | C API 实现（嵌入 .so） |
| `torch/csrc/inductor/aoti_runtime/model_container.h` | 模型容器实现 |
| `torch/csrc/inductor/aoti_runtime/model.h` | 单个模型实现 |

---

*本文基于 PyTorch 2.x 源码分析，具体实现可能随版本更新而变化。*
