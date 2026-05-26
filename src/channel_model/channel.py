"""
BSC (Binary Symmetric Channel) 和 BEC (Binary Erasure Channel) 信道仿真。

支持可调参数、随机种子复现、实际错误率统计。
"""

import random
from typing import List, Optional, Tuple, Union

from src.interfaces import Channel


class BSCChannel(Channel):
    """二进制对称信道 —— 以概率 epsilon 随机翻转每个比特。

    Args:
        epsilon: 交叉概率 (0.0 ~ 1.0)，0.0 表示无损
        seed:    随机种子，相同 seed + 相同输入 → 相同输出
    """

    def __init__(self, epsilon: float = 0.0, seed: Optional[int] = None):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon 必须在 [0, 1] 之间，收到: {epsilon}")
        self.epsilon = float(epsilon)
        self._seed = seed
        self._rng = random.Random(seed)

    def set_param(self, epsilon: float) -> None:
        """运行时动态修改交叉概率。"""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon 必须在 [0, 1] 之间，收到: {epsilon}")
        self.epsilon = float(epsilon)

    def reset(self) -> None:
        """重置随机数生成器到初始种子，便于重复实验。"""
        self._rng = random.Random(self._seed)

    def transmit(self, bits: List[int]) -> Tuple[List[int], float]:
        """
        通过 BSC 传输比特流。

        Args:
            bits: 输入比特流（每个元素为 0 或 1）

        Returns:
            (received_bits, actual_error_rate)
            - received_bits: 经过信道后的比特流
            - actual_error_rate: 实际翻转比例
        """
        n = len(bits)
        if n == 0 or self.epsilon == 0.0:
            return list(bits), 0.0

        received = []
        flipped = 0
        threshold = self.epsilon

        for b in bits:
            if self._rng.random() < threshold:
                received.append(1 - b)
                flipped += 1
            else:
                received.append(b)

        actual_rate = float(flipped) / n
        return received, actual_rate


class BECChannel(Channel):
    """二进制删除信道 —— 以概率 p 擦除每个比特（标记为 None）。

    Args:
        erasure_prob: 删除概率 (0.0 ~ 1.0)
        seed:         随机种子
    """

    def __init__(self, erasure_prob: float = 0.0, seed: Optional[int] = None):
        if not 0.0 <= erasure_prob <= 1.0:
            raise ValueError(f"erasure_prob 必须在 [0, 1] 之间，收到: {erasure_prob}")
        self.erasure_prob = float(erasure_prob)
        self._seed = seed
        self._rng = random.Random(seed)

    def set_param(self, p: float) -> None:
        """运行时动态修改删除概率。"""
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p 必须在 [0, 1] 之间，收到: {p}")
        self.erasure_prob = float(p)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def transmit(self, bits: List[int]) -> Tuple[List[Union[int, None]], float]:
        """
        通过 BEC 传输比特流。

        Args:
            bits: 输入比特流

        Returns:
            (received, actual_erasure_rate)
            - received: 混合 list，有效位为 0/1，擦除位为 None
            - actual_erasure_rate: 实际删除比例
        """
        n = len(bits)
        if n == 0 or self.erasure_prob == 0.0:
            return list(bits), 0.0

        received: List[Union[int, None]] = []
        erased = 0
        threshold = self.erasure_prob

        for b in bits:
            if self._rng.random() < threshold:
                received.append(None)
                erased += 1
            else:
                received.append(b)

        actual_rate = float(erased) / n
        return received, actual_rate


def create_channel(channel_type: str, param: float, seed: int = 42) -> Channel:
    """
    信道工厂函数，便于命令行和脚本中按字符串创建信道实例。

    Args:
        channel_type: 'bsc' 或 'bec'
        param:        信道参数（BSC 的 ε 或 BEC 的 p）
        seed:         随机种子

    Returns:
        Channel 实例
    """
    t = channel_type.strip().lower()
    if t == 'bsc':
        return BSCChannel(epsilon=param, seed=seed)
    elif t == 'bec':
        return BECChannel(erasure_prob=param, seed=seed)
    else:
        raise ValueError(f"未知信道类型: '{channel_type}'，可选 'bsc' 或 'bec'")
