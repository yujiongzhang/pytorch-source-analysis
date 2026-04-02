# ATen - 核心算子库（六）：AT_DISPATCH 宏与类型分发

> **前序**: [Part 5 - 注册机制](./05-registration.md)  
> **核心源码**: `aten/src/ATen/Dispatch.h`, `aten/src/ATen/Dispatch_v2.h`

---

## 1. 为什么需要类型分发？

### 1.1 问题背景

PyTorch 支持多种数据类型（dtype）:

```python
# 同一个算子，不同的 dtype
a_float32 = torch.tensor([1.0, 2.0])      # float32
a_int64 = torch.tensor([1, 2])            # int64
a_float16 = torch.tensor([1.0, 2.0], dtype=torch.float16)  # float16

result = torch.abs(a)  # 需要根据 dtype 调用不同的实现
```

**C++ 模板的局限**:
- 模板在编译时实例化
- PyTorch 需要在运行时根据 Tensor 的 dtype 选择实现
- 需要一种"运行时模板"机制

### 1.2 AT_DISPATCH 的作用

```
AT_DISPATCH 宏 = 运行时 switch + 编译时模板实例化

运行流程:
1. 运行时：检查 Tensor 的 ScalarType
2. Switch: 根据 ScalarType 进入对应的 case
3. 模板实例化: 在每个 case 中定义 scalar_t 类型
4. 执行：调用特化的 Kernel 代码
```

---

## 2. AT_DISPATCH 宏家族

### 2.1 宏的分类

根据支持的 dtype 组合，AT_DISPATCH 宏分为以下几类：

| 宏前缀 | 支持的 dtype | 典型用途 |
|--------|------------|---------|
| `AT_DISPATCH_ALL_TYPES` | float, double, int32, int64, int16, int8, uint8 | 通用运算 |
| `AT_DISPATCH_FLOATING_TYPES` | float, double | 浮点运算 (sin, cos) |
| `AT_DISPATCH_INTEGRAL_TYPES` | int32, int64, int16, int8, uint8 | 整数运算 |
| `AT_DISPATCH_COMPLEX_TYPES` | complex64, complex128 | 复数运算 |
| `AT_DISPATCH_QINT_TYPES` | qint8, quint8, qint32 | 量化运算 |

### 2.2 带后缀的宏

| 后缀 | 含义 | 示例 |
|------|------|------|
| `_AND_HALF` | + float16 | `AT_DISPATCH_FLOATING_TYPES_AND_HALF` |
| `_AND_BOOL` | + bool | `AT_DISPATCH_ALL_TYPES_AND_BOOL` |
| `_AND_COMPLEX` | + complex64, complex128 | `AT_DISPATCH_ALL_TYPES_AND_COMPLEX` |
| `_AND_BFLOAT16` | + bfloat16 | `AT_DISPATCH_FLOATING_TYPES_AND_BFLOAT16` |

---

## 3. 基础宏详解

### 3.1 AT_DISPATCH_FLOATING_TYPES

**源码**: `Dispatch.h` (L191-L196)

```cpp
#define AT_DISPATCH_CASE_FLOATING_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
```

**展开过程**:
```cpp
AT_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), "abs_cpu", [&]() {
    // scalar_t 被定义为当前 dtype 对应的 C++ 类型
    result = std::abs(value);
});

// 展开后类似：
switch (tensor.scalar_type()) {
  case ScalarType::Double: {
    using scalar_t = double;
    result = std::abs(value);
    break;
  }
  case ScalarType::Float: {
    using scalar_t = float;
    result = std::abs(value);
    break;
  }
  default:
    TORCH_CHECK(false, "abs_cpu not implemented for ", tensor.scalar_type());
}
```

### 3.2 AT_DISPATCH_ALL_TYPES

**源码**: `Dispatch.h` (L465-L470)

```cpp
#define AT_DISPATCH_CASE_ALL_TYPES(...)        \
  AT_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__))

// AT_DISPATCH_CASE_INTEGRAL_TYPES 展开:
#define AT_DISPATCH_CASE_INTEGRAL_TYPES(...)          \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)
```

**支持的 dtype**:
- 浮点：float, double
- 整数：uint8 (Byte), int8 (Char), int16 (Short), int32 (Int), int64 (Long)

### 3.3 AT_DISPATCH_CASE 宏

**源码**: `Dispatch.h` (L69-L71)

```cpp
#define AT_DISPATCH_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, scalar_t, __VA_ARGS__)

// 内部展开为：
case ScalarType::Float: {
    using scalar_t = float;
    __VA_ARGS__();
    break;
}
```

---

## 4. 扩展类型支持

### 4.1 半精度浮点：Half / BFloat16

**源码**: `Dispatch.h` (L198-L205)

```cpp
#define AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(__VA_ARGS__))
```

**使用场景**:
- GPU Kernel 支持 float16
- 混合精度训练

**注意**: CPU 不支持高效的 float16 计算，通常只在 GPU 上使用。

### 4.2 BFloat16 支持

**源码**: `Dispatch.h` (L207-L213)

```cpp
#define AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(...)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_REDUCED_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(__VA_ARGS__))
```

**BFloat16 vs Float16**:
- BFloat16: 1 符号位 + 8 指数位 + 7 尾数位
- Float16: 1 符号位 + 5 指数位 + 10 尾数位
- BFloat16 动态范围更大，适合深度学习训练

### 4.3 复数支持

**源码**: `Dispatch.h` (L298-L303)

```cpp
#define AT_DISPATCH_CASE_COMPLEX_TYPES(...)                    \
  AT_DISPATCH_CASE(at::ScalarType::ComplexDouble, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::ComplexFloat, __VA_ARGS__)

#define AT_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_COMPLEX_TYPES(__VA_ARGS__))
```

**复数类型**:
- `ComplexFloat` = `c10::complex<float>` (64 位)
- `ComplexDouble` = `c10::complex<double>` (128 位)

### 4.4 量化类型

**源码**: `Dispatch.h` (L472-L478)

```cpp
#define AT_DISPATCH_CASE_QINT_TYPES(...)                      \
  AT_DISPATCH_CASE_QINT(at::kQInt8, at::qint8, __VA_ARGS__)   \
  AT_DISPATCH_CASE_QINT(at::kQUInt8, at::quint8, __VA_ARGS__) \
  AT_DISPATCH_CASE_QINT(at::kQInt32, at::qint32, __VA_ARGS__)

#define AT_DISPATCH_QINT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_QINT_TYPES(__VA_ARGS__))
```

**量化类型特点**:
- 存储为 int8/uint8
- 带有 scale 和 zero_point 属性
- 用于模型量化推理

---

## 5. 添加额外类型

### 5.1 AND 系列宏

当基础宏不包含所需类型时，使用 AND 系列宏添加：

**添加单个类型**:
```cpp
#define AT_DISPATCH_CASE_FLOATING_TYPES_AND(SCALARTYPE, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

// 使用示例：
AT_DISPATCH_FLOATING_TYPES_AND(
    at::ScalarType::Half,  // 额外添加 Half
    tensor.scalar_type(), 
    "op_name", 
    [&] { /* code */ }
);
```

**添加多个类型**:
```cpp
#define AT_DISPATCH_CASE_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

// 使用示例：
AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    tensor.scalar_type(),
    "op_name",
    [&] { /* code */ }
);
```

### 5.2 组合宏示例

**浮点 + 复数 + 额外类型**:
```cpp
AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    ScalarType::Half,
    ScalarType::BFloat16,
    tensor.scalar_type(),
    "my_kernel",
    [&]() {
        // scalar_t 可以是：
        // float, double, c10::complex<float>, c10::complex<double>,
        // Half, BFloat16
    }
);
```

---

## 6. AT_DISPATCH_V2: 新版宏

### 6.1 为什么要新版本？

**旧宏的问题**:
```cpp
// 需要记住确切的宏名称
AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16, ...)
//                                                             ^^^^
//                                              需要数清楚是 AND4 还是 AND5
```

**AT_DISPATCH_V2 的优势**:
- 不需要指定 arity（AND 几个类型）
- 可以直接使用类型列表宏
- 更灵活、更易维护

### 6.2 AT_DISPATCH_V2 语法

**源码**: `Dispatch_v2.h` (L1-L90)

```cpp
// 基本语法
AT_DISPATCH_V2(
    scalar_type,          // Tensor 的 ScalarType
    "debug string",       // 调试名称（用于错误消息）
    AT_WRAP([&] {         // 必须用 AT_WRAP 包裹 lambda
        // 代码体，scalar_t 已定义
    }),
    AT_EXPAND(AT_ALL_TYPES),  // 类型列表
    kHalf,                    // 或者单个类型
    kBFloat16,
    // ... 任意数量
)
```

### 6.3 新旧对比

**旧版本**:
```cpp
AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
    kComplexHalf,
    kHalf,
    self.scalar_type(),
    "_local_scalar_dense_cpu",
    [&] {
        scalar_t value = *self.data_ptr<scalar_t>();
        r = Scalar(value);
    }
);
```

**新版本**:
```cpp
AT_DISPATCH_V2(
    self.scalar_type(),
    "_local_scalar_dense_cpu",
    AT_WRAP([&] {
        scalar_t value = *self.data_ptr<scalar_t>();
        r = Scalar(value);
    }),
    AT_EXPAND(AT_ALL_TYPES),
    AT_EXPAND(AT_COMPLEX_TYPES),
    kComplexHalf,
    kHalf
);
```

### 6.4 AT_EXPAND 用法

预定义的类型列表：

```cpp
// 所有类型（整数 + 浮点）
AT_EXPAND(AT_ALL_TYPES)
// 展开为：kByte, kChar, kInt, kLong, kShort, kDouble, kFloat

// 所有类型 + 复数
AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX)
// 展开为：AT_ALL_TYPES + kComplexDouble, kComplexFloat

// 整数类型
AT_EXPAND(AT_INTEGRAL_TYPES)
// 展开为：kByte, kChar, kInt, kLong, kShort

// 浮点类型
AT_EXPAND(AT_FLOATING_TYPES)
// 展开为：kDouble, kFloat

// Float8 类型
AT_EXPAND(AT_FLOAT8_TYPES)
// 展开为：kFloat8_e4m3fn, kFloat8_e4m3fnuz, kFloat8_e5m2, kFloat8_e5m2fnuz
```

### 6.5 实战示例

**示例 1**: fill 算子
```cpp
// ScalarOps.cpp (L39-L43)
AT_DISPATCH_V2(
    self.scalar_type(), 
    "fill_out", 
    AT_WRAP([&]() {
        fill_inplace<scalar_t>(self, value);
    }), 
    kComplexHalf, 
    kHalf, 
    kBool, 
    kBFloat16, 
    AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), 
    AT_EXPAND(AT_FLOAT8_TYPES), 
    AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES)
);
```

**示例 2**: unique 算子
```cpp
// Unique.cpp (L448)
return AT_DISPATCH_V2(
    self.scalar_type(), 
    "unique", 
    [&] AT_WRAP({
        auto [output, inverse, _] = unique_cpu_sorted_template<scalar_t>(
            self, 
            return_inverse, 
            false, 
            IsUnique<scalar_t, /* equal_nan */false>()
        );
        return std::make_tuple(output, inverse, counts);
    }),
    AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
    kBool,
    kBFloat16,
    kHalf
);
```

---

## 7. AT_DISPATCH_SWITCH

### 7.1 底层 switch 宏

**源码**: `Dispatch.h` (L183-L189)

```cpp
#define AT_DISPATCH_SWITCH(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH_TMPL(                 \
      RECORD_KERNEL_FUNCTION_DTYPE,         \
      TORCH_CHECK_NOT_IMPLEMENTED,          \
      TYPE,                                 \
      NAME,                                 \
      __VA_ARGS__)
```

### 7.2 使用场景

当需要为不同类型提供完全不同的实现时：

```cpp
AT_DISPATCH_SWITCH(tensor.scalar_type(), "my_op",
    AT_DISPATCH_CASE_FLOATING_TYPES([&] {
        // 浮点实现
        op_floating<scalar_t>(iter);
    })
    AT_DISPATCH_CASE_INTEGRAL_TYPES([&] {
        // 整数实现
        op_integral<scalar_t>(iter);
    })
    AT_DISPATCH_CASE(kBool, [&] {
        // 布尔实现
        op_bool(iter);
    })
);
```

---

## 8. 特殊类型处理

### 8.1 Bool 类型

Bool 不参与算术运算，需要特殊处理：

```cpp
// 错误：AT_DISPATCH_ALL_TYPES 不包含 Bool
AT_DISPATCH_ALL_TYPES(dtype, "add", [&] {
    // scalar_t 不会是 bool
});

// 正确：使用 AND_BOOL 宏
AT_DISPATCH_ALL_TYPES_AND_BOOL(dtype, "logical_and", [&] {
    if constexpr (std::is_same_v<scalar_t, bool>) {
        // 布尔逻辑实现
    } else {
        // 算术实现
    }
});
```

### 8.2 ComplexHalf

`ComplexHalf` 是复数半精度类型，需要特别注意：

```cpp
// ComplexHalf 的实部/虚部是 Half
// 计算时可能需要提升到 float
AT_DISPATCH_V2(
    dtype,
    "complex_op",
    AT_WRAP([&] {
        if constexpr (std::is_same_v<scalar_t, c10::complex<at::Half>>) {
            // 特殊处理 ComplexHalf
            using comp_t = c10::complex<float>;
            // 提升到 float 计算
        }
    }),
    kComplexHalf
);
```

### 8.3 Float8 类型

Float8 是新兴的低精度格式：

```cpp
AT_DISPATCH_V2(
    dtype,
    "float8_op",
    AT_WRAP([&] {
        // Float8 类型：Float8_e4m3fn, Float8_e5m2, etc.
        // 需要特殊处理 quantization
    }),
    AT_EXPAND(AT_FLOAT8_TYPES)
);
```

---

## 9. 性能优化

### 9.1 避免重复转换

```cpp
// 差：在循环内转换
AT_DISPATCH_ALL_TYPES(dtype, "op", [&]() {
    auto factor = double_value;  // 每次都是 double
    result = data * factor;      // 需要隐式转换
});

// 好：在 AT_DISPATCH 内转换
AT_DISPATCH_ALL_TYPES(dtype, "op", [&]() {
    auto factor = double_value.to<scalar_t>();  // 一次转换
    result = data * factor;  // 直接使用 scalar_t
});
```

### 9.2 使用 constexpr if

```cpp
AT_DISPATCH_V2(
    dtype,
    "optimized_op",
    AT_WRAP([&] {
        if constexpr (std::is_floating_point_v<scalar_t>) {
            // 编译时优化：浮点分支
            floating_impl<scalar_t>(data);
        } else if constexpr (std::is_integral_v<scalar_t>) {
            // 编译时优化：整数分支
            integral_impl<scalar_t>(data);
        }
    }),
    AT_EXPAND(AT_ALL_TYPES)
);
```

### 9.3 选择正确的宏

| 场景 | 推荐宏 |
|------|--------|
| 通用算术运算 | `AT_DISPATCH_ALL_TYPES` |
| 三角函数 | `AT_DISPATCH_FLOATING_TYPES` |
| 位运算 | `AT_DISPATCH_INTEGRAL_TYPES` |
| 比较运算 | `AT_DISPATCH_ALL_TYPES_AND_BOOL` |
| 复数运算 | `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES` |
| 量化推理 | `AT_DISPATCH_QINT_TYPES` |

---

## 10. 常见错误

### 10.1 错误：Unsupported dtype

```
RuntimeError: "add_cpu" not implemented for 'Bool'

原因：使用了 AT_DISPATCH_ALL_TYPES，但 Bool 不在支持列表中

解决：
// 改用：
AT_DISPATCH_ALL_TYPES_AND_BOOL(dtype, "add", [&] { ... });
```

### 10.2 错误：ZeroDivisionError

```cpp
// 整数除法未检查零
AT_DISPATCH_INTEGRAL_TYPES(dtype, "div", [&]() {
    result = a / b;  // b 可能为 0
});

// 修复：
AT_DISPATCH_INTEGRAL_TYPES(dtype, "div", [&]() {
    TORCH_CHECK(b != 0, "ZeroDivisionError");
    result = a / b;
});
```

### 10.3 错误：逗号问题

```cpp
// 错误：lambda 内有逗号时，宏解析失败
AT_DISPATCH_ALL_TYPES(dtype, "op", [&] {
    std::map<int, int> m;  // 这里的逗号会破坏宏解析
});

// 修复 1：使用 AT_WRAP
AT_DISPATCH_ALL_TYPES(dtype, "op", AT_WRAP([&] {
    std::map<int, int> m;
}));

// 修复 2：使用 AT_DISPATCH_V2
AT_DISPATCH_V2(dtype, "op", AT_WRAP([&] {
    std::map<int, int> m;
}), AT_EXPAND(AT_ALL_TYPES));
```

---

## 11. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `Dispatch.h` | L191-L196 | AT_DISPATCH_FLOATING_TYPES |
| `Dispatch.h` | L198-L205 | AT_DISPATCH_FLOATING_TYPES_AND_HALF |
| `Dispatch.h` | L298-L303 | AT_DISPATCH_COMPLEX_TYPES |
| `Dispatch.h` | L445-L453 | AT_DISPATCH_INTEGRAL_TYPES |
| `Dispatch.h` | L465-L470 | AT_DISPATCH_ALL_TYPES |
| `Dispatch.h` | L472-L478 | AT_DISPATCH_QINT_TYPES |
| `Dispatch_v2.h` | L83-L90 | AT_DISPATCH_V2 定义 |
| `Dispatch_v2.h` | L115-L150+ | AT_AP1-AT_AP36 展开宏 |

---

## 12. 下一步

| Part | 主题 |
|------|------|
| [Part 7](./07-tensor-structure.md) | Tensor 核心数据结构 |
| [Part 8](./08-autograd-integration.md) | 自动微分集成 |

---

**参考资料**:
- `aten/src/ATen/Dispatch.h` - AT_DISPATCH 宏完整定义
- `aten/src/ATen/Dispatch_v2.h` - AT_DISPATCH_V2 新实现
- `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp` - 实际使用示例
