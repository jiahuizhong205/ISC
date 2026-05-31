"""
信道编码模块：卷积码编解码器和交织器。

负责实现信道编码冗余和纠错功能，支持 BSC 和 BEC 信道。
"""

from src.channel_coding.convolutional import ConvCodec
from src.channel_coding.interleaver import BlockInterleaver, RandomInterleaver, create_interleaver

__all__ = [
    'ConvCodec',
    'BlockInterleaver',
    'RandomInterleaver',
    'create_interleaver',
]
