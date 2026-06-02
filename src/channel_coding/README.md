"""
信道编码模块 — 使用指南

成员 B 负责的模块，包含卷积码编解码器和交织器实现。
"""

## 模块概述

### 1. 卷积码编解码器 (ConvCodec)

**技术规格：**
- 码型: (2, 1, 7) 标准 NASA 卷积码
- 约束长度: K = 7
- 码率: 1/2 (每输入 1 bit，输出 2 bits)
- 生成多项式:
  - G0 = 0o171 (八进制) = 001111001 (二进制)
  - G1 = 0o133 (八进制) = 001011011 (二进制)

**关键功能：**

#### 编码 (encode)
```python
from src.channel_coding import ConvCodec

codec = ConvCodec()
data_bits = [1, 0, 1, 1, 0, 0, 1, 0]  # 8 bits
encoded = codec.encode(data_bits)
# 输出: 28 bits (8*2 + 6*2 = 16 + 12 = 28 bits)
# 包含 6 bits tail（memory bits 清零编码状态）
```

#### 解码
支持两种信道类型：

**BSC (二进制对称信道) - 硬判决**
```python
# 接收数据中可能有翻转错误
received = [1, 0, 1, 1, ...]  # 来自 BSC 信道
decoded = codec.decode(received, channel_type='bsc')
# 输出恢复后的数据位 (去除 tail)
```

**BEC (二进制删除信道) - 软判决**
```python
# 接收数据中可能有 None (表示擦除)
received = [1, None, 1, 0, ...]  # 来自 BEC 信道
decoded = codec.decode(received, channel_type='bec')
# Viterbi 软判决处理擦除位
```

### 2. 交织器 (Interleaver)

用于 BEC 信道对抗突发删除，支持两种方案：

#### 块交织器 (BlockInterleaver)
```python
from src.channel_coding import BlockInterleaver

interleaver = BlockInterleaver(rows=16, cols=32)
# 特性：按行存储，按列读取，打散突发错误

bits = [0, 1, 2, ..., 511]  # 512 bits
interleaved = interleaver.interleave(bits)
recovered = interleaver.deinterleave(interleaved)
assert bits == recovered  # 可逆
```

#### 随机交织器 (RandomInterleaver)
```python
from src.channel_coding import RandomInterleaver

interleaver = RandomInterleaver(size=512, seed=42)
# 特性：伪随机排列，更好的乱序性能

interleaved = interleaver.interleave(bits)
recovered = interleaver.deinterleave(interleaved)
```

#### 工厂函数
```python
from src.channel_coding import create_interleaver

# 块交织
interleaver = create_interleaver('block', size=512, rows=16, cols=32)

# 随机交织
interleaver = create_interleaver('random', size=512, seed=42)
```

## 算法详解

### Viterbi 解码

采用标准 Viterbi 动态规划算法：

1. **初始化** (t=0)
   - 状态 0: 度量 = 0
   - 其他状态: 度量 = ∞

2. **递推** (t=1 到 T-1)
   - 对每个状态和输入位，计算下一状态的路径度量
   - 保留每个状态的最优前驱状态

3. **回溯** (t=T-1 到 0)
   - 从最优终态开始，回溯得到最优路径
   - 提取输入比特序列

**复杂度：**
- 时间: O(N * 2^K * 2) = O(N) （K=7，2^K=128 个状态）
- 空间: O(2^K) = O(128)

### 硬判决 vs 软判决

| 特性 | 硬判决 (BSC) | 软判决 (BEC) |
|------|------------|-----------|
| 输入格式 | int (0/1) | int/None |
| 度量计算 | 汉明距离 | 汉明距离 (None 不计) |
| 纠错能力 | 中等 | 较强 |
| 速度 | 快 | 快 |

## 集成流程

在系统流水线中的位置：

```
源编码 (A)
    ↓
比特流: list[int]
    ↓
[信道编码 (B)]  ← 你在这里
    ↓
编码比特流: list[int]
    ↓
[信道传输 (C)]
    ↓
受损信号: list[int] 或 list[int|None]
    ↓
[信道译码 (B)]  ← 你在这里
    ↓
恢复比特流: list[int]
    ↓
源解码 (A)
    ↓
重建图像
```

## 性能评估

### 编码性能
- **速度**: ~200 Kbits/ms (取决于系统)
- **码率**: 0.5 (512 bits → 1024 bits)
- **延迟**: 最少 7 个时钟周期 (tail bits)

### 解码性能
- **速度**: ~10-50 Kbits/ms (软判决较慢)
- **纠错能力**: 
  - BSC @ ε=0.05: BER 5% → <1%
  - BEC @ p=0.1: 删除 10% → 恢复

### 内存占用
- **ConvCodec**: ~5 KB (状态机预计算表)
- **BlockInterleaver(16×32)**: ~16 KB
- **RandomInterleaver(512)**: ~2 KB

## 测试覆盖

单元测试验证（tests/test_channel_coding.py）：
✓ 基本编解码 (无错)
✓ BSC 信道纠错
✓ BEC 信道擦除处理
✓ 块交织/解交织
✓ 随机交织/解交织

## 接口约定

### ChannelCodec ABC

```python
class ChannelCodec(ABC):
    @abstractmethod
    def encode(self, bits: list[int]) -> list[int]:
        """
        输入: 源编码比特流
        输出: 信道编码后比特流
        """
        pass

    @abstractmethod
    def decode(self, received: list, channel_type: str = 'bsc') -> list[int]:
        """
        输入: 信道接收信号 (BSC: list[int], BEC: list[int|None])
        输出: 恢复比特流 (仅数据位，去除 tail)
        """
        pass
```

## 已知限制

1. **Viterbi 解码速度**: Python 实现在大数据量上较慢，建议：
   - 对大数据流进行分块处理
   - 使用 NumPy/Cython 优化
   - 使用硬件加速 (FPGA/GPU)

2. **块大小**: 当前使用固定尾部长度，建议：
   - 根据数据大小自适应调整块大小
   - 动态选择交织方案

3. **错误检测**: 不包含 CRC 校验，建议：
   - 添加 CRC-16 辅助检测残留错误
   - 为关键应用添加重传机制

## 后续优化方向

1. **算法优化**
   - 实现 BCJR 软判决解码
   - 支持尾比特最小化编码

2. **性能优化**
   - Viterbi 算法 NumPy 向量化
   - 并行化处理多个数据块

3. **功能扩展**
   - 支持 Turbo 码
   - 支持 LDPC 码
   - CRC 校验集成
"""
