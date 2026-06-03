"""
JPEG-like source encoder with Huffman entropy coding.

Pipeline: RGB → YCbCr → 8×8 blocks → DCT → quantization → zigzag →
         DC diff → RLE → int→bytes → Huffman → bits (0/1).
"""

import heapq
from collections import Counter
from typing import Any, Optional

import numpy as np
from scipy.fft import dct

from src.interfaces import SourceCodec

# ——— Standard JPEG quantization tables ———

_QY = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float64)

_QC = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)

# ——— Zigzag scan order for 8×8 block ———

_ZIGZAG = np.array([
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
], dtype=np.int32)

# Inverse zigzag: _ZIGZAG_INV[linear_pos] = zigzag_index
_ZIGZAG_INV = np.empty(64, dtype=np.int32)
_ZIGZAG_INV[_ZIGZAG] = np.arange(64)

# ——— YCbCr conversion matrix (ITU-R BT.601) ———

_RGB2YCBCR = np.array([
    [ 0.299,   0.587,   0.114],
    [-0.1687, -0.3313,  0.5   ],
    [ 0.5,    -0.4187, -0.0813],
], dtype=np.float64)


# =====================================================================
#  Quantization
# =====================================================================

def _build_quant_table(base: np.ndarray, quality: int) -> np.ndarray:
    """Scale a base quantization table by quality factor (1–100)."""
    if quality >= 50:
        scale = (100 - quality) / 50
    else:
        scale = 50 / quality
    scaled = base * scale
    scaled = np.clip(np.round(scaled), 1, 255)
    return scaled.astype(np.float64)


# =====================================================================
#  Colour-space conversion
# =====================================================================

def _rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB (H, W, 3) to float64 YCbCr (H, W, 3), values in [0, 255]."""
    ycbcr = image.astype(np.float64) @ _RGB2YCBCR.T
    ycbcr[:, :, [1, 2]] += 128.0
    return ycbcr


# =====================================================================
#  Padding & block iteration
# =====================================================================

def _pad_to_multiple(image: np.ndarray, multiple: int = 8) -> tuple[np.ndarray, int, int]:
    """Symmetric-pad image so H, W are multiples of *multiple*.

    Returns (padded, orig_h, orig_w).
    """
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if image.ndim == 3:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='symmetric')
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='symmetric')
    return padded, h, w


def _iter_blocks(image: np.ndarray, block_size: int = 8):
    """Yield (i, j, c, block) for each non-overlapping block_size×block_size tile.

    *image* shape is (H, W, C); blocks are yielded per-channel.
    """
    h, w = image.shape[:2]
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            for c in range(image.shape[2]):
                yield i, j, c, image[i:i+block_size, j:j+block_size, c]


# =====================================================================
#  DCT
# =====================================================================

def _dct2d(block: np.ndarray) -> np.ndarray:
    """2-D DCT-II (orthonormal) on an 8×8 block via scipy."""
    return dct(dct(block.T, type=2, norm='ortho').T, type=2, norm='ortho')


# =====================================================================
#  Zigzag
# =====================================================================

def _zigzag(block: np.ndarray) -> np.ndarray:
    """Re-order an 8×8 block into a 64-element 1-D zigzag array."""
    return block.flat[_ZIGZAG]


def _inverse_zigzag(coeffs_1d: np.ndarray) -> np.ndarray:
    """Convert a 64-element zigzag-ordered array back to an 8×8 block."""
    block_flat = coeffs_1d[_ZIGZAG_INV]
    return block_flat.reshape(8, 8)


# =====================================================================
#  Run-length encoding / decoding
# =====================================================================

def _rle_encode(coeffs: np.ndarray) -> list[int]:
    """Run-length encode the zigzag-ordered coefficients.

    Returns a flat list: ``[dc, run₀, val₀, run₁, val₁, …]`` terminated by
    an EOB marker ``(0, 0)`` when trailing coefficients are all zero.
    A block whose last AC coefficient (index 63) is non-zero omits EOB.
    """
    dc = int(round(coeffs[0]))
    out = [dc]

    run = 0
    for val in coeffs[1:]:
        v = int(round(val))
        if v == 0:
            run += 1
            continue
        out.append(run)
        out.append(v)
        run = 0

    if run > 0:
        out.append(0)
        out.append(0)

    return out


def _rle_decode(rle_flat: list[int], idx: int) -> tuple[np.ndarray, int]:
    """Decode one block from the flat RLE list starting at *idx*.

    Returns ``(coeffs_1d, next_idx)`` where *coeffs_1d* is a 64-element
    zigzag-ordered array.
    """
    coeffs = np.zeros(64, dtype=np.float64)
    dc = rle_flat[idx]
    idx += 1
    coeffs[0] = float(dc)

    k = 1  # next AC position in zigzag order
    while k < 64 and idx + 1 < len(rle_flat):
        run = rle_flat[idx]
        val = rle_flat[idx + 1]
        idx += 2
        if run == 0 and val == 0:
            break                     # EOB

        # 容错：run 值明显异常时停止该块解码
        if run < 0 or run > 63 or k + run >= 64:
            break
        k += run
        coeffs[k] = float(val)
        k += 1

    return coeffs, idx


# =====================================================================
#  Integer ↔ bytes  (signed 16-bit big-endian)
# =====================================================================

def _ints_to_bytes(ints: list[int]) -> bytes:
    """Pack a list of ints as signed 16-bit big-endian bytes."""
    return np.array(ints, dtype='>i2').tobytes()


def _bytes_to_ints(data: bytes) -> list[int]:
    """Unpack signed 16-bit big-endian bytes back to a list of ints.

    Truncates trailing odd byte that may result from corrupted Huffman decode.
    """
    if len(data) % 2 != 0:
        data = data[:-1]
    if len(data) == 0:
        return []
    return np.frombuffer(data, dtype='>i2').tolist()


# =====================================================================
#  Huffman encoder
# =====================================================================

class _Node:
    __slots__ = ('freq', 'symbol', 'left', 'right')

    def __init__(self, freq: int, symbol: Optional[int] = None,
                 left: Optional['_Node'] = None, right: Optional['_Node'] = None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other: '_Node') -> bool:
        return self.freq < other.freq


class _Huffman:
    """Build Huffman codes from byte frequencies and encode a byte string to bits."""

    def __init__(self):
        self.codes: dict[int, str] = {}   # byte (0-255) → bit string

    def build(self, data: bytes) -> None:
        """Build Huffman code table from byte frequencies in *data*."""
        if not data:
            return
        freqs = Counter(data)

        if len(freqs) == 1:
            sym = next(iter(freqs))
            self.codes[sym] = '0'
            return

        heap = [_Node(freq, sym) for sym, freq in freqs.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            heapq.heappush(heap, _Node(a.freq + b.freq, left=a, right=b))

        self._walk(heap[0], '')

    def _walk(self, node: _Node, prefix: str) -> None:
        if node.symbol is not None:
            self.codes[node.symbol] = prefix or '0'
        else:
            self._walk(node.left, prefix + '0')
            self._walk(node.right, prefix + '1')

    def encode(self, data: bytes) -> list[int]:
        """Encode *data* bytes to a list of 0/1 ints."""
        bits: list[int] = []
        for b in data:
            code = self.codes[b]
            bits.extend(int(ch) for ch in code)
        return bits

    @property
    def table(self) -> dict[str, str]:
        """Return code table as ``{symbol_hex: bit_string}`` for header serialisation."""
        return {f'{sym:02x}': code for sym, code in self.codes.items()}

    @staticmethod
    def decode_table(raw: dict[str, str]) -> dict[str, str]:
        """Reverse *raw* table → ``{bit_string: symbol_hex}`` for the decoder."""
        return {code: sym for sym, code in raw.items()}


# =====================================================================
#  JPEG Encoder  (SourceCodec interface)
# =====================================================================

class DCTEncoder(SourceCodec):
    """JPEG-like encoder implementing ``SourceCodec``.

    Parameters
    ----------
    quality : int
        JPEG quality factor 1–100 (default 75).  Higher = better quality /
        weaker quantization.
    """

    def __init__(self, quality: int = 75, repeat: int = 5):
        if not 1 <= quality <= 100:
            raise ValueError("quality must be in [1, 100]")
        self.quality = quality
        self.repeat = repeat
        self._QY = _build_quant_table(_QY, quality)
        self._QC = _build_quant_table(_QC, quality)

    # —————————————————————— SourceCodec interface ——————————————————————

    def encode(self, image: np.ndarray) -> dict:
        """Lossy source encoding.

        Args:
            image: (H, W, 3) uint8 RGB image.

        Returns:
            dict::

                {
                    'bits': list[int],   # 0/1 bitstream (Huffman-encoded)
                    'header': {
                        'orig_h': int,
                        'orig_w': int,
                        'channels': 3,
                        'quality': int,
                        'huffman_table': {hex_sym: bit_string, …},
                    }
                }
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be (H, W, 3) uint8 RGB")
        orig_h, orig_w = image.shape[:2]

        # 1. RGB → YCbCr
        ycbcr = _rgb_to_ycbcr(image)

        # 2. Pad to multiples of 8 (symmetric)
        padded, _, _ = _pad_to_multiple(ycbcr)

        # 3. Block-wise: DCT → quantize → zigzag → RLE
        #    每块的 RLE 数据单独编码，block_lengths 记录 Huffman 后字节数
        block_rle: list[list[int]] = []

        for _, _, c, block in _iter_blocks(padded):
            dct_block = _dct2d(block.astype(np.float64) - 128.0)
            qtable = self._QY if c == 0 else self._QC
            quantized = np.round(dct_block / qtable)
            zigzagged = _zigzag(quantized)
            block_rle.append(_rle_encode(zigzagged))

        # 4. 每块 RLE → int16 bytes → 全局 Huffman → bits（定长补到字节边界）
        all_block_bytes: list[bytes] = []
        all_raw = bytearray()
        for rle in block_rle:
            raw = _ints_to_bytes(rle)
            all_block_bytes.append(raw)
            all_raw.extend(raw)

        huff = _Huffman()
        huff.build(bytes(all_raw))  # 用所有块的数据建全局 Huffman 表

        bits = []
        block_byte_lengths: list[int] = []
        for raw in all_block_bytes:
            block_bits = huff.encode(raw)
            # 补到整字节
            pad = (8 - len(block_bits) % 8) % 8
            if pad:
                block_bits.extend([0] * pad)
            # Nx 重复：单 bit 错只毁本块，多数投票消除残留错误
            for _ in range(self.repeat):
                bits.extend(block_bits)
            block_byte_lengths.append((len(block_bits) // 8) * self.repeat)

        header = {
            'orig_h': orig_h,
            'orig_w': orig_w,
            'channels': 3,
            'quality': self.quality,
            'block_byte_lengths': block_byte_lengths,
            'huffman_table': huff.table,
            'repeat': self.repeat,
        }

        return {'bits': bits, 'header': header}

    def decode(self, bits: list[int], header: dict) -> Any:
        """Decode — implemented in ``decoder.py``."""
        raise NotImplementedError("decoder is in decoder.py")
