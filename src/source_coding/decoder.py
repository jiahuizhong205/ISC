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
        """Walk the prefix tree bit-by-bit, emitting a byte on each leaf."""
        result = bytearray()
        node = self._root
        for b in bits:
            ch = str(b)
            if ch not in node:
                raise ValueError(
                    f"Bit {b} at position {len(result)} does not match "
                    f"any Huffman prefix (partial decode: {bytes(result)!r})"
                )
            node = node[ch]
            if '_sym' in node:
                result.append(node['_sym'])
                node = self._root
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

class JPEGDecoder(SourceCodec):
    """JPEG-like decoder implementing ``SourceCodec``.

    Reverses the pipeline produced by ``JPEGEncoder``: Huffman-decode the
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
            bits: 0/1 bitstream from ``JPEGEncoder.encode``.
            header: dict with keys ``orig_h``, ``orig_w``, ``quality``,
                    ``huffman_table``.

        Returns:
            (H, W, 3) uint8 RGB reconstructed image.
        """
        orig_h: int = header['orig_h']
        orig_w: int = header['orig_w']
        quality: int = header['quality']
        huffman_table: dict[str, str] = header['huffman_table']

        # Padded dimensions (reverse of symmetric pad)
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        padded_h = orig_h + pad_h
        padded_w = orig_w + pad_w

        # Rebuild quantisation tables
        QY = _build_quant_table(_QY, quality)
        QC = _build_quant_table(_QC, quality)

        # 1. Huffman decode: bits → bytes
        huff = _HuffmanDecoder(huffman_table)
        raw_bytes = huff.decode(bits)

        # 2. Bytes → RLE integer list
        rle_flat = _bytes_to_ints(raw_bytes)

        # 3. Block-by-block reconstruction
        ycbcr = np.empty((padded_h, padded_w, 3), dtype=np.float64)
        prev_dc = [0.0, 0.0, 0.0]   # Y, Cb, Cr
        idx = 0

        for i in range(0, padded_h, 8):
            for j in range(0, padded_w, 8):
                for c in range(3):
                    coeffs_1d, idx = _rle_decode(rle_flat, idx)

                    # DC differential recovery
                    coeffs_1d[0] += prev_dc[c]
                    prev_dc[c] = coeffs_1d[0]

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
