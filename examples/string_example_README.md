# StringType for HierarchicalKV

这个示例演示如何使用自定义`StringType`作为HierarchicalKV哈希表中的值。

## 概述

在`include/nv/string_type.cuh`中的`StringType`类提供了一个兼容CUDA的固定大小字符串实现，可以用作HierarchicalKV哈希表中的值类型(V)。它支持基本的字符串操作，如连接、子字符串提取和比较。

## 特性

- 兼容CUDA的固定大小缓冲区字符串类
- 可自定义最大字符串长度的模板参数
- 支持标准字符串操作
- 预定义别名的可配置字符串大小：`String`（256字符）、`SmallString`（64字符）和`LargeString`（1024字符）
- 与HierarchicalKV哈希表作为值类型兼容

## 使用StringType

以下是使用`StringType`类的方法：

1. 包含头文件：
```cpp
#include "nv/string_type.cuh"
```

2. 使用预定义的字符串别名或使用自定义大小创建自己的：
```cpp
using namespace nv;

// 使用预定义别名
String default_str = "Hello, world!";               // 最大256字符
SmallString small_str = "Short text";               // 最大64字符
LargeString large_str = "Very long text...";        // 最大1024字符

// 或定义自定义大小
StringType<512> custom_str = "Custom sized string"; // 最大512字符
```

3. 在HierarchicalKV哈希表中使用字符串类型作为值：
```cpp
using namespace nv;
using namespace nv::merlin;

using K = uint64_t;               // 键类型
using V = String;                 // 值类型（256字符的字符串）
using S = uint64_t;               // 分数类型
using StringHashTable = HashTable<K, V, S, EvictStrategy::kLru>;
```

## 示例代码

`string_example.cu`文件演示了如何：

1. 定义和初始化带有`StringType`值的HierarchicalKV哈希表
2. 生成并插入字符串值
3. 通过键查找字符串值
4. 处理字符串的批处理操作

## 字符串操作

`StringType`类支持以下操作：

- 字符串赋值：`str = "new text";`
- 连接：`str1 + str2`或`str1 += str2`
- 字符访问：`str[index]`
- 比较：`str1 == str2`、`str1 != str2`等
- 长度查询：`str.length()`、`str.empty()`
- 子字符串提取：`str.substr(pos, len)`
- 字符串搜索：`str.find("substring")`或`str.find('c')`

## 性能考虑

- `StringType`使用固定大小的缓冲区，使其适合无需动态内存分配的GPU操作
- 使用最小适当大小以获得更好的内存效率
- 对于大字符串，考虑使用`LargeString`或指定自定义大小
- 批处理操作比单个操作更高效

## 编译和运行

编译示例：

```bash
nvcc -o string_example examples/string_example.cu -I./include -lcuda
```

运行示例：

```bash
./string_example
``` 