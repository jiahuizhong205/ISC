"""
交织器（Interleaver）：用于 BEC 信道对抗突发删除。

交织将连续的比特流按特定排列重新组织，使得删除簇在交织后分散开来，
有利于后续的信道编码纠错。

支持多种交织方案：块交织、随机交织等。
"""

import random
from typing import List, Optional, Tuple


class BlockInterleaver:
    """
    块交织器：将比特流按行存储、按列读取。

    方式：
      输入：b0 b1 b2 b3 b4 b5 ... (按行存储)
      矩阵：
          b0  b1  ...
          ... ... ...
      输出：按列读取 (交织后)

    优点：简单，对突发错误有一定防御能力。
    缺点：延迟为 rows * cols
    """

    def __init__(self, rows: int = 16, cols: int = 32):
        """
        初始化块交织器。

        Args:
            rows: 矩阵行数
            cols: 矩阵列数
        """
        self.rows = rows
        self.cols = cols
        self.block_size = rows * cols

    def interleave(self, bits: List[int]) -> List[int]:
        """
        交织：比特流按行存入矩阵，按列读出。

        Args:
            bits: 输入比特流

        Returns:
            交织后的比特流
        """
        interleaved: List[int] = []

        for start_idx in range(0, len(bits), self.block_size):
            block = bits[start_idx : start_idx + self.block_size]

            # 按行填充矩阵
            matrix = [
                block[i * self.cols : (i + 1) * self.cols]
                for i in range(self.rows)
            ]

            # 确保每行长度为 cols
            for i in range(self.rows):
                if len(matrix[i]) < self.cols:
                    matrix[i].extend([0] * (self.cols - len(matrix[i])))

            # 按列读出
            for col in range(self.cols):
                for row in range(self.rows):
                    if col < len(matrix[row]):
                        interleaved.append(matrix[row][col])

        return interleaved

    def deinterleave(self, bits: List[int]) -> List[int]:
        """
        解交织：反向操作。

        Args:
            bits: 交织后的比特流

        Returns:
            原始比特流顺序
        """
        deinterleaved: List[int] = []

        for start_idx in range(0, len(bits), self.block_size):
            block = bits[start_idx : start_idx + self.block_size]

            # 按列存入矩阵
            matrix = [[0] * self.cols for _ in range(self.rows)]
            idx = 0
            for col in range(self.cols):
                for row in range(self.rows):
                    if idx < len(block):
                        matrix[row][col] = block[idx]
                        idx += 1

            # 按行读出
            for row in range(self.rows):
                for col in range(self.cols):
                    deinterleaved.append(matrix[row][col])

        return deinterleaved


class RandomInterleaver:
    """
    随机交织器：使用伪随机排列。

    方式：
      生成长度为 N 的随机排列 π
      输出[i] = 输入[π[i]]

    优点：对抗任意错误模式，接近理论最优。
    缺点：需要保存排列表。
    """

    def __init__(self, size: int = 512, seed: Optional[int] = None):
        """
        初始化随机交织器。

        Args:
            size: 交织块大小
            seed: 随机种子，便于复现
        """
        self.size = size
        self.seed = seed
        self._rng = random.Random(seed)
        self._permutation = list(range(size))
        self._rng.shuffle(self._permutation)

    def interleave(self, bits: List[int]) -> List[int]:
        """
        按随机排列交织。

        Args:
            bits: 输入比特流

        Returns:
            交织后的比特流
        """
        interleaved: List[int] = []

        for start_idx in range(0, len(bits), self.size):
            block = bits[start_idx : start_idx + self.size]
            # 补齐到 size
            while len(block) < self.size:
                block.append(0)

            # 按排列重新组织
            reordered = [0] * self.size
            for i, perm_idx in enumerate(self._permutation):
                if i < len(block):
                    reordered[perm_idx] = block[i]

            interleaved.extend(reordered)

        return interleaved

    def deinterleave(self, bits: List[int]) -> List[int]:
        """
        反向排列。

        Args:
            bits: 交织后的比特流

        Returns:
            原始比特流顺序
        """
        deinterleaved: List[int] = []

        for start_idx in range(0, len(bits), self.size):
            block = bits[start_idx : start_idx + self.size]
            while len(block) < self.size:
                block.append(0)

            # 反向排列
            reordered = [0] * self.size
            for i, perm_idx in enumerate(self._permutation):
                if perm_idx < len(block):
                    reordered[i] = block[perm_idx]

            deinterleaved.extend(reordered)

        return deinterleaved


def create_interleaver(
    interleaver_type: str = 'block',
    size: int = 512,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    seed: Optional[int] = None,
) -> object:
    """
    工厂函数：创建指定类型的交织器。

    Args:
        interleaver_type: 'block' 或 'random'
        size:             总大小
        rows:             块交织的行数
        cols:             块交织的列数
        seed:             随机交织的种子

    Returns:
        交织器实例
    """
    if interleaver_type.lower() == 'block':
        if rows is None:
            rows = 16
        if cols is None:
            cols = size // rows
        return BlockInterleaver(rows, cols)
    elif interleaver_type.lower() == 'random':
        return RandomInterleaver(size, seed)
    else:
        raise ValueError(f"未知交织类型: {interleaver_type}")
