"""
端到端系统流水线：读图 → 源编码 → 信道编码 → 信道传输 → 信道译码 → 源解码 → 输出。

支持模块渐进接入：当 A/B 的模块未就绪时自动降级为直通模式，
信道模型 (C) 可独立运行和验证。

用法:
    python src/main.py --image data/kodak/kodim01.png --channel bsc --param 0.05
    python src/main.py --image data/kodak/kodim01.png --channel bec --param 0.1 --quality 30
    python src/main.py --image data/kodak/kodim01.png --channel bsc --param 0.0 --save-bin output/test.bin
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from src.channel_model.channel import create_channel
from src.interfaces import Channel


# ═══════════════════════════════════════════════════════════════════════
# 命令行参数
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ISC Project 2 — 图像编码传输流水线'
    )
    parser.add_argument(
        '--image', type=str, required=True,
        help='输入图像路径'
    )
    parser.add_argument(
        '--channel', type=str, default='bsc', choices=['bsc', 'bec'],
        help='信道类型: bsc (默认) 或 bec'
    )
    parser.add_argument(
        '--param', type=float, default=0.0,
        help='BSC 交叉概率 ε 或 BEC 删除概率 p (默认 0.0 = 无损)'
    )
    parser.add_argument(
        '--quality', type=int, default=50,
        help='源编码 quality factor 1-100 (默认 50)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子 (默认 42)'
    )
    parser.add_argument(
        '--output', type=str, default='output/recovered.png',
        help='输出重建图像路径 (默认 output/recovered.png)'
    )
    parser.add_argument(
        '--save-bin', type=str, default=None,
        help='保存压缩比特流到 .bin 文件'
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# 图像 I/O
# ═══════════════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    """加载 RGB 图像为 uint8 numpy array。"""
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.uint8)


def save_image(arr: np.ndarray, path: str) -> None:
    """保存 numpy array 为 PNG 图像。"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    clipped = arr.clip(0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(path)


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)。

    PSNR = 10 * log10(MAX² / MSE)，MAX = 255 (8-bit 图像)
    """
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = np.mean(diff ** 2)
    if mse == 0.0:
        return float('inf')
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def image_to_raw_bits(img: np.ndarray) -> List[int]:
    """将图像像素展平并转换为比特流（无压缩，用于降级模式）。"""
    flattened = img.flatten()
    bits: List[int] = []
    for val in flattened:
        for shift in range(7, -1, -1):
            bits.append((int(val) >> shift) & 1)
    return bits


def raw_bits_to_image(bits: List[int], shape: Tuple[int, ...]) -> np.ndarray:
    """将原始比特流恢复为图像（image_to_raw_bits 的逆操作）。"""
    pixels_per_channel = len(bits) // 8
    values: List[int] = []
    for i in range(pixels_per_channel):
        val = 0
        for j in range(8):
            idx = i * 8 + j
            val = (val << 1) | (bits[idx] if idx < len(bits) else 0)
        values.append(val)

    expected_pixels = 1
    for s in shape:
        expected_pixels *= s
    values = values[:expected_pixels]
    return np.array(values, dtype=np.uint8).reshape(shape)


# ═══════════════════════════════════════════════════════════════════════
# 主流水线
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("  ISC Project 2 — 图像编码传输流水线")
    print("=" * 50)
    print(f"  输入图像:    {args.image}")
    print(f"  信道类型:    {args.channel.upper()}")
    print(f"  信道参数:    {args.param}")
    print(f"  源编码 Q:    {args.quality}")
    print(f"  随机种子:    {args.seed}")
    print()

    # ── 1. 加载图像 ──
    t_total = time.time()
    img = load_image(args.image)
    print(f"[1/6] 图像加载     {img.shape}  {img.dtype}")

    # ── 2. 源编码 ──
    source_available = False
    try:
        from src.source_coding.encoder import DCTEncoder

        t0 = time.time()
        encoder = DCTEncoder(quality=args.quality)
        source_bits, header = encoder.encode(img)
        t_enc = time.time() - t0
        source_available = True
        ratio = img.size * 8 / len(source_bits) if source_bits else float('inf')
        print(f"[2/6] 源编码       {len(source_bits)} bits  "
              f"压缩率 {ratio:.1f}x  耗时 {t_enc:.3f}s")
    except ImportError:
        print(f"[2/6] 源编码       [跳过] 模块未就绪，使用原始像素比特")
        source_bits = image_to_raw_bits(img)
        header = {'shape': list(img.shape), 'fallback': True}
        t_enc = 0.0

    # ── 3. 信道编码 ──
    channel_codec = None
    try:
        from src.channel_coding.convolutional import ConvCodec

        t0 = time.time()
        channel_codec = ConvCodec()
        encoded_bits = channel_codec.encode(source_bits)
        t_ch_enc = time.time() - t0
        print(f"[3/6] 信道编码     {len(source_bits)} -> {len(encoded_bits)} bits  "
              f"码率 {len(source_bits)/len(encoded_bits):.2f}  耗时 {t_ch_enc:.3f}s")
    except ImportError:
        print(f"[3/6] 信道编码     [跳过] 模块未就绪，直通模式")
        encoded_bits = list(source_bits)
        t_ch_enc = 0.0

    # ── 4. 信道传输 ──
    channel: Channel = create_channel(args.channel, args.param, seed=args.seed)
    t0 = time.time()
    received, actual_error_rate = channel.transmit(encoded_bits)
    t_tx = time.time() - t0
    print(f"[4/6] 信道传输     {args.channel.upper()} param={args.param}  "
          f"实际错误率 {actual_error_rate:.4f}  耗时 {t_tx:.3f}s")

    # ── 5. 信道译码 ──
    try:
        from src.channel_coding.convolutional import ConvCodec

        t0 = time.time()
        if channel_codec is None:
            channel_codec = ConvCodec()
        decoded_bits = channel_codec.decode(received, channel_type=args.channel)
        t_ch_dec = time.time() - t0

        # 统计 BER 改善
        raw_errors = sum(
            1 for a, r in zip(source_bits, received)
            if r is not None and a != r
        )
        final_errors = sum(
            1 for a, d in zip(source_bits, decoded_bits)
            if a != d
        )
        print(f"[5/6] 信道译码     Viterbi 译码  "
              f"BER: {actual_error_rate:.4f} -> {final_errors/len(source_bits):.4f}  "
              f"耗时 {t_ch_dec:.3f}s")
    except ImportError:
        # BEC: 擦除位填 0 保持长度；BSC: 直接传递
        erased_count = sum(1 for b in received if b is None)
        decoded_bits = [int(b) if b is not None else 0 for b in received]
        t_ch_dec = 0.0
        if erased_count > 0:
            print(f"[5/6] 信道译码     [降级] 模块未就绪，擦除位填0 "
                  f"(共 {erased_count}/{len(received)} 位)")
        else:
            print(f"[5/6] 信道译码     [跳过] 模块未就绪，直通模式")
        t_ch_dec = 0.0

    # ── 6. 源解码 ──
    try:
        from src.source_coding.decoder import DCTDecoder

        t0 = time.time()
        decoder = DCTDecoder()
        recovered = decoder.decode(decoded_bits, header)
        t_dec = time.time() - t0
        print(f"[6/6] 源解码       耗时 {t_dec:.3f}s")
    except ImportError:
        if header.get('fallback'):
            t0 = time.time()
            recovered = raw_bits_to_image(decoded_bits, tuple(header['shape']))
            t_dec = time.time() - t0
        else:
            print(f"[6/6] 源解码       [失败] 模块未就绪且非降级模式，无法重建")
            recovered = img
            t_dec = 0.0
        print(f"[6/6] 源解码       [降级] 比特流直接还原  耗时 {t_dec:.3f}s")

    # ── 结果 ──
    t_total = time.time() - t_total
    save_image(recovered, args.output)
    psnr = compute_psnr(img, recovered)

    print()
    print("  " + "-" * 40)
    print(f"  PSNR:        {psnr:.2f} dB")
    print(f"  总耗时:      {t_total:.3f}s")
    print(f"  输出图像:    {args.output}")
    print("  " + "-" * 40)

    # 可选：保存比特流
    if args.save_bin:
        from src.bitstream import pack_bitstream
        bin_data = pack_bitstream(decoded_bits, header)
        os.makedirs(os.path.dirname(args.save_bin) or '.', exist_ok=True)
        with open(args.save_bin, 'wb') as f:
            f.write(bin_data)
        print(f"  比特流:      {args.save_bin}  ({len(bin_data)} bytes)")


if __name__ == '__main__':
    main()
