"""
卷积码编解码器：(2, 1, 7) 卷积码 + Viterbi 硬/软判决译码。

支持 BSC (二进制对称信道) 和 BEC (二进制删除信道) 信道适配。
- BSC: 接收比特可能翻转，使用硬判决 Viterbi
- BEC: 接收可能为 None (擦除)，使用软判决 Viterbi
"""

import struct
from typing import List, Optional, Tuple, Union
import numpy as np

from src.interfaces import ChannelCodec


class ConvCodec(ChannelCodec):
    """
    (2, 1, 7) 卷积码编解码器。

    标准 NASA code:
      G0 = 171 (octal) = 001111001 (binary)
      G1 = 133 (octal) = 001011011 (binary)

    约束长度 K=7，码率 1/2。
    """

    def __init__(self, poly0: int = 0o171, poly1: int = 0o133, k: int = 7):
        """
        初始化卷积码编码器。

        Args:
            poly0: 第一个生成多项式（八进制）
            poly1: 第二个生成多项式（八进制）
            k:     约束长度
        """
        self.poly0 = poly0
        self.poly1 = poly1
        self.k = k
        self.memory = k - 1

        # 预计算状态转移表
        self._build_state_machine()

    def _build_state_machine(self) -> None:
        """预计算状态机：每个状态在输入 0/1 时的转移和输出。"""
        num_states = 1 << self.memory

        # next_state[state][input] -> next_state
        self.next_state = [[0, 0] for _ in range(num_states)]
        # output[state][input] -> (out0, out1)
        self.output = [[(0, 0), (0, 0)] for _ in range(num_states)]

        for state in range(num_states):
            for inp in [0, 1]:
                # 状态寄存器左移，新bit进入最低位
                shift_reg = (state << 1) | inp

                # 输出 1：parity of poly0
                out0 = self._parity(shift_reg & self.poly0)
                # 输出 2：parity of poly1
                out1 = self._parity(shift_reg & self.poly1)

                # 下一个状态：去掉最高位
                next_s = (shift_reg >> 1) & ((1 << self.memory) - 1)

                self.next_state[state][inp] = next_s
                self.output[state][inp] = (out0, out1)

    @staticmethod
    def _parity(val: int) -> int:
        """计算奇偶性（0 个或偶数个 1 为 0，奇数个 1 为 1）。"""
        count = 0
        while val:
            count += val & 1
            val >>= 1
        return count & 1

    def encode(self, bits: List[int]) -> List[int]:
        """
        卷积码编码：输入每 1 bit 产生 2 bit 输出。

        Args:
            bits: 源编码比特流

        Returns:
            编码后的比特流 (长度约为输入的 2 倍)
        """
        encoded: List[int] = []
        state = 0

        for bit in bits:
            out0, out1 = self.output[state][bit]
            encoded.append(out0)
            encoded.append(out1)
            state = self.next_state[state][bit]

        # 尾部填充：将状态寄存器清零 (memory bits)
        for _ in range(self.memory):
            out0, out1 = self.output[state][0]
            encoded.append(out0)
            encoded.append(out1)
            state = self.next_state[state][0]

        return encoded

    def decode(self, received: list, channel_type: str = 'bsc') -> List[int]:
        """
        Viterbi 解码。

        Args:
            received: 信道输出
                - BSC: list[int]，含翻转错误
                - BEC: list[int|None]，None 表示擦除
            channel_type: 'bsc' 或 'bec'

        Returns:
            译码比特流
        """
        if channel_type.lower() == 'bsc':
            return self._viterbi_hard(received)
        elif channel_type.lower() == 'bec':
            return self._viterbi_soft(received)
        else:
            raise ValueError(f"未知信道类型: {channel_type}")

    def _viterbi_hard(self, received: List[int]) -> List[int]:
        """
        硬判决 Viterbi：将接收比特作为汉明距离计算度量。

        Args:
            received: 受损编码比特流 (0 或 1)

        Returns:
            译码后的数据比特流（去除尾部tail位）
        """
        num_states = 1 << self.memory
        num_symbols = len(received) // 2

        # path_metric[state] = 到当前时刻的最小路径度量
        path_metric = [float('inf')] * num_states
        path_metric[0] = 0.0

        # 记录最优路径上的输入比特，用于回溯
        path_history = [[0] * num_states for _ in range(num_symbols)]

        for t in range(num_symbols):
            r0 = received[2 * t]
            r1 = received[2 * t + 1]

            new_path_metric = [float('inf')] * num_states

            for prev_state in range(num_states):
                if path_metric[prev_state] == float('inf'):
                    continue

                for inp in [0, 1]:
                    next_state = self.next_state[prev_state][inp]
                    out0, out1 = self.output[prev_state][inp]

                    # 汉明距离：接收与输出的汉明距离
                    dist = (r0 != out0) + (r1 != out1)
                    new_metric = path_metric[prev_state] + dist

                    if new_metric < new_path_metric[next_state]:
                        new_path_metric[next_state] = new_metric
                        path_history[t][next_state] = inp

            path_metric = new_path_metric

        # 回溯：从最优终态开始反向追踪
        decoded = []
        state = 0
        for t in range(num_symbols - 1, -1, -1):
            bit = path_history[t][state]
            decoded.append(bit)
            # 反向更新状态
            for prev_state in range(1 << self.memory):
                if self.next_state[prev_state][bit] == state:
                    state = prev_state
                    break

        decoded.reverse()

        # 去除尾部tail位（编码时添加的memory位）
        data_bits = num_symbols - self.memory
        return decoded[:data_bits]

    def _viterbi_soft(self, received: List[Union[int, None]]) -> List[int]:
        """
        软判决 Viterbi：处理擦除 (None)。

        对于 BEC，None 表示信道不确定，我们给予最大度量值。

        Args:
            received: 混合列表，int 表示有效比特，None 表示擦除

        Returns:
            译码后的数据比特流（去除尾部tail位）
        """
        # 先清理 received：对于 None，选择某个默认值或维持不确定状态
        num_symbols = len(received) // 2

        path_metric = [0.0] * (1 << self.memory)
        path_history = [[0] * (1 << self.memory) for _ in range(num_symbols)]

        for t in range(num_symbols):
            r0 = received[2 * t]
            r1 = received[2 * t + 1]

            new_path_metric = [float('inf')] * (1 << self.memory)

            for prev_state in range(1 << self.memory):
                for inp in [0, 1]:
                    next_state = self.next_state[prev_state][inp]
                    out0, out1 = self.output[prev_state][inp]

                    # 计算软度量：None 不增加度量，有效比特按汉明距离计算
                    cost = 0.0
                    if r0 is not None and r0 != out0:
                        cost += 1.0
                    if r1 is not None and r1 != out1:
                        cost += 1.0

                    new_metric = path_metric[prev_state] + cost

                    if new_metric < new_path_metric[next_state]:
                        new_path_metric[next_state] = new_metric
                        path_history[t][next_state] = inp

            path_metric = new_path_metric

        # 回溯
        decoded = []
        state = 0
        for t in range(num_symbols - 1, -1, -1):
            bit = path_history[t][state]
            decoded.append(bit)
            for prev_state in range(1 << self.memory):
                if self.next_state[prev_state][bit] == state:
                    state = prev_state
                    break

        decoded.reverse()

        # 去除尾部tail位（编码时添加的memory位）
        data_bits = num_symbols - self.memory
        return decoded[:data_bits]
