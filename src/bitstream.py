"""
比特流打包/拆包工具。

将源编码产出的 meta 信息（图像尺寸、Q值、Huffman码表等）
与压缩比特流打包为单一二进制文件，接收端按协议解析恢复。

二进制格式:
    [4 bytes: header_json_len, big-endian uint32]
    [header_json_len bytes: UTF-8 JSON]
    [remaining bytes: 比特流，每 8 bit 打包为 1 byte，末尾不足补 0]
"""

import json
import struct
from typing import Any, Dict, List, Tuple


def pack_bitstream(bits: List[int], header: Dict[str, Any]) -> bytes:
    """
    将 header(JSON) 和比特流打包为二进制数据。

    Args:
        bits:   比特流 list[int]，每个元素 0 或 1
        header: 解码所需的元信息字典（尺寸、码表、Q值等）

    Returns:
        bytes: 可写入 .bin 文件的二进制数据

    Example:
        >>> data = pack_bitstream([1,0,1,1], {'shape': [64,64,3]})
        >>> bits2, hdr2, _ = unpack_bitstream(data)
        >>> bits2 == [1,0,1,1] and hdr2 == {'shape': [64,64,3]}
        True
    """
    # 复制 header，避免修改调用方的字典
    header = dict(header)

    # 补齐到 8 的倍数，记录填充位数
    original_len = len(bits)
    padding = (8 - original_len % 8) % 8
    padded_bits = list(bits)
    if padding > 0:
        padded_bits.extend([0] * padding)

    # 将 padding 记入 header（必须在 JSON 序列化之前）
    header['_padding'] = padding

    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header_len = len(header_json)

    payload = bytearray()
    payload.extend(struct.pack('>I', header_len))
    payload.extend(header_json)

    # 比特打包为字节
    for i in range(0, len(padded_bits), 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | padded_bits[i + j]
        payload.append(byte_val)

    return bytes(payload)


def unpack_bitstream(data: bytes) -> Tuple[List[int], Dict[str, Any], int]:
    """
    从二进制数据中解析出比特流和 header。

    Args:
        data: pack_bitstream 产出的 bytes

    Returns:
        (bits, header, padding_count)
        - bits:   解析出的比特流
        - header: 元信息字典
        - padding_count: 末尾被填充的比特个数
    """
    if len(data) < 4:
        raise ValueError("数据太短，无法解析 header 长度")

    header_len = struct.unpack('>I', data[:4])[0]

    if len(data) < 4 + header_len:
        raise ValueError(
            f"数据不完整：期望 {4 + header_len} bytes 用于 header，实际 {len(data)} bytes"
        )

    header_json = data[4:4 + header_len].decode('utf-8')
    header = json.loads(header_json)

    padding = header.pop('_padding', 0)

    payload = data[4 + header_len:]
    bits_raw: List[int] = []
    for byte in payload:
        for j in range(8):
            bits_raw.append((byte >> (7 - j)) & 1)

    # 丢弃填充位
    if padding > 0:
        bits_raw = bits_raw[:-padding]

    return bits_raw, header, padding


def bits_to_bytes(bits: List[int]) -> Tuple[bytes, int]:
    """
    纯比特流转 bytes（不含 header），末尾补 0 至整字节。

    Returns:
        (bytes_data, original_bit_length)
    """
    n = len(bits)
    padded = list(bits)
    pad = (8 - n % 8) % 8
    if pad > 0:
        padded.extend([0] * pad)

    out = bytearray()
    for i in range(0, len(padded), 8):
        val = 0
        for j in range(8):
            val = (val << 1) | padded[i + j]
        out.append(val)

    return bytes(out), n


def bytes_to_bits(data: bytes, original_length: int) -> List[int]:
    """
    将 bits_to_bytes 产出的 bytes 恢复为指定长度的比特流。

    Args:
        data:             字节数据
        original_length:  原始比特流长度（不含填充）

    Returns:
        list[int]: 恢复后的比特流
    """
    bits: List[int] = []
    for byte in data:
        for j in range(8):
            bits.append((byte >> (7 - j)) & 1)
    return bits[:original_length]
