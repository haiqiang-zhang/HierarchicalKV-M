# CuString Python Bindings

## Overview / 概述

CuString is a simple, fixed-size string type designed for CUDA compatibility and use with the Merlin hash table. This document describes the Python bindings for CuString.

CuString 是一个简单的固定大小字符串类型，专为 CUDA 兼容性和 Merlin 哈希表使用而设计。本文档描述了 CuString 的 Python 绑定。

## Features / 特性

- **Fixed Size**: Maximum length of 10 characters (including null terminator)
- **CUDA Compatible**: Can be used in both host and device code
- **Python Integration**: Full Python bindings with intuitive interface
- **Memory Efficient**: Minimal memory footprint

- **固定大小**: 最大长度 10 个字符（包括空终止符）
- **CUDA 兼容**: 可在主机和设备代码中使用
- **Python 集成**: 完整的 Python 绑定，界面直观
- **内存高效**: 最小内存占用

## Installation / 安装

Make sure the `merlin_hashtable_python` module is built and available in your Python path.

确保 `merlin_hashtable_python` 模块已构建并在您的 Python 路径中可用。

## Usage / 使用方法

### Basic Usage / 基本用法

```python
import merlin_hashtable_python as mht

# Create an empty CuString / 创建空的 CuString
s1 = mht.CuString()
print(f"Empty string: '{s1}'")  # Empty string: ''

# Create from string literal / 从字符串字面量创建
s2 = mht.CuString("Hello")
print(f"Hello string: '{s2}'")  # Hello string: 'Hello'

# Copy constructor / 拷贝构造函数
s3 = mht.CuString(s2)
print(f"Copied string: '{s3}'")  # Copied string: 'Hello'
```

### String Operations / 字符串操作

```python
# Assignment / 赋值
s = mht.CuString()
s.assign("World")
print(f"Assigned: '{s}'")  # Assigned: 'World'

# Length / 长度
print(f"Length: {len(s)}")  # Length: 5

# Character access / 字符访问
print(f"First char: '{s[0]}'")  # First char: 'W'
print(f"Last char: '{s[4]}'")   # Last char: 'd'

# Character modification / 字符修改
s[0] = 'w'
print(f"Modified: '{s}'")  # Modified: 'world'
```

### String Comparison / 字符串比较

```python
s1 = mht.CuString("Hello")
s2 = mht.CuString("Hello")
s3 = mht.CuString("World")

# Equality / 相等性
print(s1 == s2)        # True
print(s1 == "Hello")   # True
print(s1 == s3)        # False
```

### String Concatenation / 字符串连接

```python
s = mht.CuString("Hi")
print(f"Before: '{s}'")  # Before: 'Hi'

# In-place addition / 就地加法
s += " there"
print(f"After: '{s}'")   # After: 'Hi there'

# Note: Result will be truncated if it exceeds max length
# 注意：如果超过最大长度，结果将被截断
```

### Utility Methods / 实用方法

```python
s = mht.CuString("Test")

# Check if empty / 检查是否为空
print(s.empty())  # False

# Get C-style string / 获取 C 风格字符串
print(s.c_str())  # Test

# Clear the string / 清空字符串
s.clear()
print(s.empty())  # True
```

### Properties / 属性

```python
# Maximum length / 最大长度
print(mht.CuString.max_length)  # 10

# Size in bytes / 字节大小
print(mht.CuString.sizeof)      # 10
```

## Limitations / 限制

1. **Fixed Size**: Maximum 10 characters (including null terminator)
2. **Truncation**: Strings longer than 9 characters will be truncated
3. **No Dynamic Allocation**: Memory is statically allocated

1. **固定大小**: 最多 10 个字符（包括空终止符）
2. **截断**: 超过 9 个字符的字符串将被截断
3. **无动态分配**: 内存是静态分配的

## Example: Long String Handling / 示例：长字符串处理

```python
# Long string will be truncated / 长字符串将被截断
long_str = "This is a very long string"
s = mht.CuString(long_str)
print(f"Result: '{s}'")     # Result: 'This is a'
print(f"Length: {len(s)}")  # Length: 9
```

## Integration with Hash Table / 与哈希表集成

CuString can be used as a value type in the Merlin hash table:

CuString 可以用作 Merlin 哈希表中的值类型：

```python
# Example usage with hash table (pseudo-code)
# 与哈希表的示例用法（伪代码）
options = mht.HashTableOptions()
options.dim = mht.CuString.sizeof  # Set dimension to CuString size

# Use CuString as value type in hash table operations
# 在哈希表操作中使用 CuString 作为值类型
```

## Error Handling / 错误处理

```python
try:
    s = mht.CuString("Hello")
    # Index out of range will raise exception
    # 索引超出范围将引发异常
    char = s[20]  # Raises IndexError
except IndexError as e:
    print(f"Index error: {e}")
```

## Best Practices / 最佳实践

1. **Check Length**: Always be aware of the 10-character limit
2. **Handle Truncation**: Consider truncation when working with user input
3. **Use for Short Strings**: Best suited for short identifiers, codes, etc.

1. **检查长度**: 始终注意 10 个字符的限制
2. **处理截断**: 处理用户输入时考虑截断
3. **用于短字符串**: 最适合短标识符、代码等

## Testing / 测试

Run the test script to verify functionality:

运行测试脚本以验证功能：

```bash
python test_custring.py
```

This will test all CuString operations and report any issues.

这将测试所有 CuString 操作并报告任何问题。 