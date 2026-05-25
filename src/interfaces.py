"""
模块接口抽象基类。

成员 A 的 source_coding 和成员 B 的 channel_coding 需继承对应接口实现，
确保系统集成时无需修改流水线代码。
"""

from abc import ABC, abstractmethod
from typing import Any


class SourceCodec(ABC):
    """源编解码器接口"""

    @abstractmethod
    def encode(self, image: Any) -> dict:
        """
        有损源编码。

        Args:
            image: numpy array (H, W, 3) RGB 图像, dtype=uint8

        Returns:
            dict: {
                'bits': list[int],       # 压缩后的比特流
                'header': dict,          # 解码所需元信息（尺寸、码表等）
            }
        """
        ...

    @abstractmethod
    def decode(self, bits: list[int], header: dict) -> Any:
        """
        源解码。

        Args:
            bits: 信道译码恢复后的比特流
            header: 编码时产出的元信息

        Returns:
            numpy array (H, W, 3) RGB 重建图像, dtype=uint8
        """
        ...


class ChannelCodec(ABC):
    """信道编解码器接口"""

    @abstractmethod
    def encode(self, bits: list[int]) -> list[int]:
        """
        信道编码：为源编码比特流添加冗余。

        Args:
            bits: 源编码输出的比特流

        Returns:
            list[int]: 编码后的比特流
        """
        ...

    @abstractmethod
    def decode(self, received: list, channel_type: str = 'bsc') -> list[int]:
        """
        信道译码：从受损信号中恢复原始比特。

        Args:
            received: 信道输出的信号
                - BSC: list[int]，含翻转错误
                - BEC: list，有效位为 int，丢失位为 None
            channel_type: 'bsc' 或 'bec'

        Returns:
            list[int]: 译码修正后的比特流
        """
        ...


class Channel(ABC):
    """信道模型接口"""

    @abstractmethod
    def transmit(self, bits: list[int]) -> tuple[list, float]:
        """
        通过信道传输比特流。

        Args:
            bits: 输入比特流

        Returns:
            tuple:
                - received: 受损信号 (BSC: list[int] / BEC: list[int|None])
                - actual_rate: 实际误码率/删除率
        """
        ...
