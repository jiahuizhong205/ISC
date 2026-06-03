#!/usr/bin/env python
"""
批量实验脚本 —— 遍历图集 × 信道类型 × 参数组合，一键运行并导出 CSV。

用法:
    python scripts/run_experiments.py                       # 跑全部实验
    python scripts/run_experiments.py --subset 3             # 只用前 3 张图快速验证
    python scripts/run_experiments.py --channels bsc         # 只跑 BSC
    python scripts/run_experiments.py --csv results/test.csv # 指定输出路径
"""

import argparse
import csv
import os
import sys
import time
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.channel_model.channel import create_channel


# ═══════════════════════════════════════════════════════════════════════
# 实验参数空间
# ═══════════════════════════════════════════════════════════════════════
BSC_PARAMS = [0.0, 0.01, 0.05, 0.1]
BEC_PARAMS = [0.0, 0.05, 0.1, 0.2]
Q_VALUES   = [10, 50, 90]
IMAGE_DIR  = 'data/kodak'
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ISC Project 2 — 批量实验')
    parser.add_argument('--subset', type=int, default=0,
                        help='只用前 N 张图像 (0=全部)')
    parser.add_argument('--channels', type=str, default='both',
                        choices=['bsc', 'bec', 'both'],
                        help='只跑指定信道类型 (默认 both)')
    parser.add_argument('--csv', type=str, default='results/experiments.csv',
                        help='CSV 输出路径 (默认 results/experiments.csv)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='随机种子 (默认 42)')
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

def discover_images(subset: int = 0) -> Dict[str, np.ndarray]:
    """
    扫描图像目录，返回 {文件名: numpy_array}。

    优先查找 data/kodak/，若不存在则查找 data/ 下的任意 PNG。
    """
    search_dirs = [IMAGE_DIR, 'data']
    images: Dict[str, np.ndarray] = {}

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith('.png'))
        if subset and subset < len(files):
            files = files[:subset]
        for f in files:
            path = os.path.join(d, f)
            try:
                from PIL import Image
                img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
                images[f] = img
            except Exception as e:
                print(f"  [WARN] 跳过 {path}: {e}")
        if images:
            break

    return images


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """逐像素 PSNR (dB)。"""
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = np.mean(diff ** 2)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def image_to_raw_bits(img: np.ndarray) -> List[int]:
    """图像像素展平为原始比特流（降级模式）。"""
    bits: List[int] = []
    for val in img.flatten():
        for shift in range(7, -1, -1):
            bits.append((int(val) >> shift) & 1)
    return bits


def raw_bits_to_image(bits: List[int], shape: Tuple[int, ...]) -> np.ndarray:
    """原始比特流恢复为图像。"""
    needed = 1
    for s in shape:
        needed *= s
    values: List[int] = []
    for i in range(min(needed, len(bits) // 8)):
        val = 0
        for j in range(8):
            idx = i * 8 + j
            val = (val << 1) | (bits[idx] if idx < len(bits) else 0)
        values.append(val)
    while len(values) < needed:
        values.append(0)
    return np.array(values, dtype=np.uint8).reshape(shape)


# ═══════════════════════════════════════════════════════════════════════
# 单次实验
# ═══════════════════════════════════════════════════════════════════════

def run_single(image_name: str, original: np.ndarray,
               channel_type: str, param: float,
               quality: int, seed: int) -> Dict[str, Any]:
    """
    执行单次端到端实验，返回结果字典。

    当 A/B 模块未就绪时自动降级，保证信道模型能独立验证。
    """
    result: Dict[str, Any] = {
        'image': image_name,
        'channel': channel_type,
        'param': param,
        'quality': quality,
        'psnr': None,
        'time_source_enc': None,
        'time_channel_enc': None,
        'time_transmission': None,
        'time_channel_dec': None,
        'time_source_dec': None,
        'compression_ratio': None,
        'actual_error_rate': None,
    }

    # ── 源编码 ──
    header: Dict[str, Any] = {}
    try:
        from src.source_coding.encoder import DCTEncoder
        t0 = time.time()
        encoder = DCTEncoder(quality=quality, repeat=5)
        encoded = encoder.encode(original); source_bits = encoded['bits']; header = encoded['header']
        result['time_source_enc'] = time.time() - t0
        result['compression_ratio'] = (
            original.size * 8 / len(source_bits)
            if source_bits else float('inf')
        )
    except ImportError:
        source_bits = image_to_raw_bits(original)
        header = {'shape': list(original.shape), 'fallback': True}
        result['time_source_enc'] = 0.0
        result['compression_ratio'] = 1.0

    # ── 信道编码 ──
    channel_codec = None
    try:
        from src.channel_coding.convolutional import ConvCodec
        t0 = time.time()
        channel_codec = ConvCodec()
        encoded_bits = channel_codec.encode(source_bits)
        result['time_channel_enc'] = time.time() - t0
    except ImportError:
        encoded_bits = list(source_bits)
        result['time_channel_enc'] = 0.0

    # ── 交织 ──
    interleaver = None
    encoded_len_before_interleave = len(encoded_bits)
    try:
        from src.channel_coding.interleaver import BlockInterleaver
        t0 = time.time()
        interleaver = BlockInterleaver(rows=64, cols=128)
        tx_bits = interleaver.interleave(encoded_bits)
        result['time_interleave'] = time.time() - t0
    except ImportError:
        tx_bits = list(encoded_bits)
        result['time_interleave'] = 0.0

    # ── 信道传输 ──
    channel = create_channel(channel_type, param, seed=seed)
    t0 = time.time()
    received, actual_rate = channel.transmit(tx_bits)
    result['time_transmission'] = time.time() - t0
    result['actual_error_rate'] = actual_rate

    # ── 解交织 + 信道译码 ──
    try:
        from src.channel_coding.convolutional import ConvCodec

        t0 = time.time()
        if interleaver is not None:
            deinterleaved = interleaver.deinterleave(received)
            deinterleaved = deinterleaved[:encoded_len_before_interleave]
        else:
            deinterleaved = list(received)
        t_deinter = time.time() - t0

        t0 = time.time()
        if channel_codec is None:
            channel_codec = ConvCodec()
        decoded_bits = channel_codec.decode(deinterleaved, channel_type=channel_type)
        result['time_channel_dec'] = time.time() - t0 + t_deinter
    except ImportError:
        if interleaver is not None:
            received = interleaver.deinterleave(received)[:encoded_len_before_interleave]
        decoded_bits = [int(b) if b is not None else 0 for b in received]
        result['time_channel_dec'] = 0.0

    # ── 源解码 ──
    try:
        from src.source_coding.decoder import DCTDecoder
        t0 = time.time()
        decoder = DCTDecoder()
        recovered = decoder.decode(decoded_bits, header)
        result['time_source_dec'] = time.time() - t0
    except ImportError:
        if header.get('fallback'):
            t0 = time.time()
            recovered = raw_bits_to_image(decoded_bits, tuple(header['shape']))
            result['time_source_dec'] = time.time() - t0
        else:
            recovered = original
            result['time_source_dec'] = 0.0

    # ── PSNR ──
    result['psnr'] = compute_psnr(original, recovered)
    return result


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # 发现图像
    images = discover_images(subset=args.subset)
    if not images:
        print("[INFO] 未找到图集，生成随机测试图 (64x64x3)")
        rng = np.random.RandomState(args.seed)
        images = {'random_test.png': rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)}

    # 构建实验列表
    channels_to_run = ['bsc', 'bec'] if args.channels == 'both' else [args.channels]

    experiments: List[Dict[str, Any]] = []
    total = len(channels_to_run) * (len(BSC_PARAMS) + len(BEC_PARAMS)) * len(Q_VALUES) * len(images)
    count = 0

    print(f"开始批量实验: {len(images)} 张图像, {len(channels_to_run)} 种信道, "
          f"共计约 {total} 组")
    print("-" * 50)

    for ch_type in channels_to_run:
        params = BSC_PARAMS if ch_type == 'bsc' else BEC_PARAMS
        for param, q in product(params, Q_VALUES):
            for name, img in images.items():
                count += 1
                result = run_single(name, img, ch_type, param, q, args.seed)
                experiments.append(result)
                if count % 10 == 0 or count == total:
                    print(f"  [{count}/{total}] {name:20s}  "
                          f"{ch_type.upper()} p={param:.2f} Q={q:2d}  "
                          f"PSNR={result['psnr']:.1f} dB")

    # ── 写入 CSV ──
    os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
    fieldnames = [
        'image', 'channel', 'param', 'quality', 'psnr',
        'time_source_enc', 'time_channel_enc', 'time_transmission',
        'time_channel_dec', 'time_source_dec', 'compression_ratio',
        'actual_error_rate',
    ]
    with open(args.csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(experiments)

    # ── 汇总统计 ──
    psnrs = [e['psnr'] for e in experiments
             if e['psnr'] is not None and e['psnr'] != float('inf')]
    print()
    print("=" * 50)
    print(f"  实验完成: {len(experiments)} 组")
    if psnrs:
        print(f"  PSNR 范围: {min(psnrs):.1f} ~ {max(psnrs):.1f} dB  "
              f"(均值 {np.mean(psnrs):.1f})")
    print(f"  结果保存:  {args.csv}")
    print("=" * 50)


if __name__ == '__main__':
    main()
