"""
JPEG-like source decoder.

Pipeline: bits → Huffman decode → bytes → ints → RLE decode → inverse zigzag
         → inverse quantize → inverse DCT → merge blocks → YCbCr→RGB → crop.
"""

from typing import Any

import numpy as np
from scipy.fft import idct

from src.interfaces import SourceCodec
from src.source_coding.encoder import (
    _bytes_to_ints,
    _build_quant_table,
    _inverse_zigzag,
    _rle_decode,
    _QY,
    _QC,
    _RGB2YCBCR,
)

# Pre-compute inverse colour matrix once at import time
_YCBCR2RGB = np.linalg.inv(_RGB2YCBCR)


# =====================================================================
#  Huffman decoder
# =====================================================================

class _HuffmanDecoder:
    """Decode a Huffman-encoded bitstream back to bytes."""

    def __init__(self, table: dict[str, str]):
        """Build a prefix tree from *table* (``{hex_sym: bit_string}``)."""
        self._root: dict[str, dict] = {}
        for hex_sym, code in table.items():
            node = self._root
            for ch in code:
                node = node.setdefault(ch, {})
            node['_sym'] = int(hex_sym, 16)

    def decode(self, bits: list[int]) -> bytes:
        """Walk the prefix tree bit-by-bit, emitting a byte on each leaf.

        Corrupted bits that don't match any prefix cause a reset to root,
        allowing the decoder to resynchronise on subsequent bits.
        """
        result = bytearray()
        node = self._root
        errors = 0
        for b in bits:
            ch = str(b)
            if ch not in node:
                # 非法码字：bit 损坏导致路径不存在，重置继续
                errors += 1
                node = self._root
                # 重试当前 bit 在新 root 下
                if ch in node:
                    node = node[ch]
                continue
            node = node[ch]
            if '_sym' in node:
                result.append(node['_sym'])
                node = self._root
        if errors > 0:
            import sys
            print(f"  [WARN] Huffman 解码: {errors} 个非法码字被跳过 "
                  f"(共 {len(bits)} bits)", file=sys.stderr)
        return bytes(result)


# =====================================================================
#  Inverse DCT
# =====================================================================

def _idct2d(block: np.ndarray) -> np.ndarray:
    """2-D inverse DCT-II (orthonormal) on an 8×8 block via scipy."""
    return idct(idct(block.T, type=2, norm='ortho').T, type=2, norm='ortho')


# =====================================================================
#  Inverse colour-space conversion
# =====================================================================

def _ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """Convert float64 YCbCr (H, W, 3) → uint8 RGB (H, W, 3)."""
    ycbcr = ycbcr.copy()
    ycbcr[:, :, [1, 2]] -= 128.0
    rgb = ycbcr @ _YCBCR2RGB.T
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


# =====================================================================
#  JPEG Decoder  (SourceCodec interface)
# =====================================================================

class DCTDecoder(SourceCodec):
    """JPEG-like decoder implementing ``SourceCodec``.

    Reverses the pipeline produced by ``DCTEncoder``: Huffman-decode the
    bitstream, unpack RLE symbols, rebuild coefficient blocks, apply
    inverse DCT, and convert back to RGB.
    """

    # —————————————————————— SourceCodec interface ——————————————————————

    def encode(self, image: np.ndarray) -> dict:
        """Encode — implemented in ``encoder.py``."""
        raise NotImplementedError("encoder is in encoder.py")

    def decode(self, bits: list[int], header: dict) -> np.ndarray:
        """Lossy source decoding.

        Args:
            bits: 0/1 bitstream from ``DCTEncoder.encode``.
            header: dict with keys ``orig_h``, ``orig_w``, ``quality``,
                    ``huffman_table``.

        Returns:
            (H, W, 3) uint8 RGB reconstructed image.
        """
        orig_h: int = header['orig_h']
        orig_w: int = header['orig_w']
        quality: int = header['quality']

        # Padded dimensions (reverse of symmetric pad)
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        padded_h = orig_h + pad_h
        padded_w = orig_w + pad_w

        # Rebuild quantisation tables
        QY = _build_quant_table(_QY, quality)
        QC = _build_quant_table(_QC, quality)

        # 1. bits → bytes
        byte_count = len(bits) // 8
        raw_bytes_all = bytes(
            sum(bits[i * 8 + j] << (7 - j) for j in range(8))
            for i in range(byte_count)
        )

        # 2. 每块独立解码——按 block_byte_lengths 跳块 + Nx 多数投票 + Huffman
        block_byte_lengths: list[int] = header.get('block_byte_lengths', [])
        huffman_table: dict[str, str] = header.get('huffman_table', {})
        repeat: int = header.get('repeat', 1)
        ycbcr = np.empty((padded_h, padded_w, 3), dtype=np.float64)

        # 预计算每个块的字节偏移
        offsets = [0]
        for bl in block_byte_lengths:
            offsets.append(offsets[-1] + bl)

        block_idx = 0
        for i in range(0, padded_h, 8):
            for j in range(0, padded_w, 8):
                for c in range(3):
                    coeffs_1d = np.zeros(64, dtype=np.float64)
                    if block_idx < len(block_byte_lengths):
                        byte_start = offsets[block_idx]
                        byte_end = offsets[block_idx + 1]
                        if byte_start < len(raw_bytes_all):
                            try:
                                block_bytes = raw_bytes_all[byte_start:min(byte_end, len(raw_bytes_all))]

                                if huffman_table and repeat > 1:
                                    # 多数投票 + Huffman 解码
                                    chunk = len(block_bytes) // repeat
                                    # 逐 byte 投票
                                    voted = bytearray()
                                    for pos in range(chunk):
                                        val = 0
                                        for bit_pos in range(8):
                                            ones = 0
                                            for r in range(repeat):
                                                idx = pos + r * chunk
                                                if idx < len(block_bytes):
                                                    ones += (block_bytes[idx] >> (7 - bit_pos)) & 1
                                            val = (val << 1) | (1 if ones > repeat // 2 else 0)
                                        voted.append(val)
                                    voted_bits = []
                                    for b in voted:
                                        for s in range(7, -1, -1):
                                            voted_bits.append((b >> s) & 1)
                                    huff_dec = _HuffmanDecoder(huffman_table)
                                    block_raw_bytes = huff_dec.decode(voted_bits)
                                    block_ints = _bytes_to_ints(block_raw_bytes)
                                    coeffs_1d, _ = _rle_decode(block_ints, 0)
                                elif huffman_table:
                                    # Huffman 解码 (无重复)
                                    block_bits = []
                                    for b in block_bytes:
                                        for s in range(7, -1, -1):
                                            block_bits.append((b >> s) & 1)
                                    huff_dec = _HuffmanDecoder(huffman_table)
                                    block_raw_bytes = huff_dec.decode(block_bits)
                                    block_ints = _bytes_to_ints(block_raw_bytes)
                                    coeffs_1d, _ = _rle_decode(block_ints, 0)
                                else:
                                    # 无 Huffman 表：直接 int16 解码
                                    block_ints = _bytes_to_ints(block_bytes)
                                    coeffs_1d, _ = _rle_decode(block_ints, 0)
                            except Exception:
                                pass  # 块损坏，保持零块（灰色）
                        block_idx += 1
                    # block_idx 超出范围：保持零块

                    block = _inverse_zigzag(coeffs_1d)          # 8×8
                    qtable = QY if c == 0 else QC
                    block *= qtable                             # inverse quantisation
                    block = _idct2d(block)                      # inverse DCT
                    block += 128.0                              # level-shift back

                    ycbcr[i:i + 8, j:j + 8, c] = block

        # 4. YCbCr → RGB
        rgb = _ycbcr_to_rgb(ycbcr)

        # 5. Crop padding
        return rgb[:orig_h, :orig_w, :]
