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
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from src.channel_model.channel import create_channel
from src.interfaces import Channel


# ═══════════════════════════════════════════════════════════════════════
# 参数加载：配置文件 + 命令行合并 (命令行优先级更高)
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ISC Project 2 — 图像编码传输流水线'
    )
    parser.add_argument(
        '--config', type=str, default='config.json',
        help='配置文件路径 (默认 config.json，不存在则忽略)'
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help='输入图像路径'
    )
    parser.add_argument(
        '--channel', type=str, default=None, choices=['bsc', 'bec'],
        help='信道类型: bsc 或 bec'
    )
    parser.add_argument(
        '--param', type=float, default=None,
        help='BSC 交叉概率 ε 或 BEC 删除概率 p'
    )
    parser.add_argument(
        '--quality', type=int, default=None,
        help='源编码 quality factor 1-100'
    )
    parser.add_argument(
        '--repeat', type=int, default=None,
        help='重复编码次数 1/3/5 (1=最快 5=最强)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='随机种子'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='输出重建图像路径'
    )
    parser.add_argument(
        '--save-bin', type=str, default=None,
        help='保存压缩比特流到 .bin 文件'
    )
    cli = parser.parse_args()

    # 从配置文件加载默认值
    config = {}
    if os.path.isfile(cli.config):
        with open(cli.config, 'r', encoding='utf-8') as f:
            config = json.load(f)

    # 合并：命令行优先级高于配置文件
    defaults = {
        'image': 'data/kodim01.png',
        'channel': 'bsc',
        'param': 0.0,
        'quality': 50,
        'repeat': 5,
        'seed': 42,
        'output': 'output/result.png',
        'save_bin': None,
    }
    for key in defaults:
        # 配置文件值
        if key not in defaults or defaults[key] is None:
            if key in config:
                defaults[key] = config[key]
    # 配置文件覆盖
    for key in ('image', 'channel', 'param', 'quality', 'repeat', 'seed', 'output', 'save_bin'):
        if key in config and cli.__dict__.get(key) is None:
            defaults[key] = config[key]
    # 命令行覆盖
    for key in ('image', 'channel', 'param', 'quality', 'repeat', 'seed', 'output', 'save_bin'):
        val = cli.__dict__.get(key)
        if val is not None:
            defaults[key] = val

    # 必填参数校验
    if not defaults.get('image'):
        parser.error('必须指定 --image 或在配置文件中设置 image')

    return argparse.Namespace(**defaults)


# ═══════════════════════════════════════════════════════════════════════
# 图像 I/O
# ═══════════════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    """加载 RGB 图像为 uint8 numpy array。"""
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.uint8)


def save_image(arr: np.ndarray, path: str) -> str:
    """保存 numpy array 为 PNG 图像。同名文件自动加 _1, _2 序号。

    Returns:
        实际使用的文件路径
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    # 同名时自动加序号
    base, ext = os.path.splitext(path)
    final_path = path
    counter = 1
    while os.path.exists(final_path):
        final_path = f"{base}_{counter}{ext}"
        counter += 1

    clipped = arr.clip(0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(final_path)

    if final_path != path:
        print(f"  已存在同名文件，保存为: {final_path}")
    return final_path


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
    print(f"  重复次数:    {args.repeat}")
    print(f"  随机种子:    {args.seed}")
    print()

    # 评估指标变量初始化
    final_errors = 0
    t_deinter = 0.0
    t_inter = 0.0

    # ── 1. 加载图像 ──
    t_total = time.time()
    img = load_image(args.image)
    print(f"[1/7] 图像加载     {img.shape}  {img.dtype}")

    # ── 2. 源编码 ──
    source_available = False
    try:
        from src.source_coding.encoder import DCTEncoder

        t0 = time.time()
        encoder = DCTEncoder(quality=args.quality, repeat=args.repeat)
        encoded = encoder.encode(img)
        source_bits = encoded['bits']
        header = encoded['header']
        t_enc = time.time() - t0
        source_available = True
        ratio = img.size * 8 / len(source_bits) if source_bits else float('inf')
        print(f"[2/7] 源编码       {len(source_bits)} bits  "
              f"压缩率 {ratio:.1f}x  耗时 {t_enc:.3f}s")
    except ImportError:
        print(f"[2/7] 源编码       [跳过] 模块未就绪，使用原始像素比特")
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
        print(f"[3/7] 信道编码     {len(source_bits)} -> {len(encoded_bits)} bits  "
              f"码率 {len(source_bits)/len(encoded_bits):.2f}  耗时 {t_ch_enc:.3f}s")
    except ImportError:
        print(f"[3/7] 信道编码     [跳过] 模块未就绪，直通模式")
        encoded_bits = list(source_bits)
        t_ch_enc = 0.0

    # ── 4. 交织 ──
    interleaver = None
    encoded_len_before_interleave = len(encoded_bits)
    try:
        from src.channel_coding.interleaver import BlockInterleaver

        t0 = time.time()
        interleaver = BlockInterleaver(rows=64, cols=128)
        tx_bits = interleaver.interleave(encoded_bits)
        t_inter = time.time() - t0
        print(f"[4/7] 交织         {len(encoded_bits)} bits  "
              f"块大小 {interleaver.block_size}  耗时 {t_inter:.3f}s")
    except ImportError:
        print(f"[4/7] 交织         [跳过] 模块未就绪")
        tx_bits = list(encoded_bits)
        t_inter = 0.0

    # ── 5. 信道传输 ──
    channel: Channel = create_channel(args.channel, args.param, seed=args.seed)
    t0 = time.time()
    received, actual_error_rate = channel.transmit(tx_bits)
    t_tx = time.time() - t0
    print(f"[5/7] 信道传输     {args.channel.upper()} param={args.param}  "
          f"实际错误率 {actual_error_rate:.4f}  耗时 {t_tx:.3f}s")

    # ── 6. 解交织 + 信道译码 ──
    try:
        from src.channel_coding.convolutional import ConvCodec

        # 解交织：先恢复比特顺序再译码
        t0 = time.time()
        if interleaver is not None:
            # BEC 的 None 值在解交织时只是被重新排列位置
            deinterleaved = interleaver.deinterleave(received)
            # 截断交织时补的填充位
            deinterleaved = deinterleaved[:encoded_len_before_interleave]
        else:
            deinterleaved = list(received)
        t_deinter = time.time() - t0

        t0 = time.time()
        if channel_codec is None:
            channel_codec = ConvCodec()
        decoded_bits = channel_codec.decode(deinterleaved, channel_type=args.channel)
        t_ch_dec = time.time() - t0

        # 统计 BER 改善
        final_errors = sum(
            1 for a, d in zip(source_bits, decoded_bits)
            if a != d
        )
        print(f"[6/7] 解交织+译码  Viterbi 译码  "
              f"BER: {final_errors/len(source_bits):.4f}  "
              f"耗时 {t_deinter + t_ch_dec:.3f}s  "
              f"(解交织 {t_deinter:.3f}s + 译码 {t_ch_dec:.3f}s)")
    except ImportError:
        # 解交织（即使信道译码模块未就绪）
        if interleaver is not None:
            received = interleaver.deinterleave(received)[:encoded_len_before_interleave]
        # BEC: 擦除位填 0 保持长度；BSC: 直接传递
        erased_count = sum(1 for b in received if b is None)
        decoded_bits = [int(b) if b is not None else 0 for b in received]
        final_errors = sum(1 for a, d in zip(source_bits, decoded_bits) if a != d)
        t_ch_dec = 0.0
        if erased_count > 0:
            print(f"[6/7] 解交织+译码  [降级] 模块未就绪，擦除位填0 "
                  f"(共 {erased_count}/{len(received)} 位)")
        else:
            print(f"[6/7] 解交织+译码  [跳过] 模块未就绪，直通模式")

    # ── 6. 源解码 ──
    try:
        from src.source_coding.decoder import DCTDecoder

        t0 = time.time()
        decoder = DCTDecoder()
        recovered = decoder.decode(decoded_bits, header)
        t_dec = time.time() - t0
        print(f"[7/7] 源解码       耗时 {t_dec:.3f}s")
    except ImportError:
        if header.get('fallback'):
            t0 = time.time()
            recovered = raw_bits_to_image(decoded_bits, tuple(header['shape']))
            t_dec = time.time() - t0
        else:
            print(f"[7/7] 源解码       [失败] 模块未就绪且非降级模式，无法重建")
            recovered = img
            t_dec = 0.0
        print(f"[7/7] 源解码       [降级] 比特流直接还原  耗时 {t_dec:.3f}s")

    # ── 评估指标 ──
    t_total = time.time() - t_total
    saved_path = save_image(recovered, args.output)
    psnr = compute_psnr(img, recovered)

    # 准确率计算
    ch_errors = int(actual_error_rate * len(encoded_bits))  # 信道造成的编码比特错误数
    viterbi_ber = final_errors / len(source_bits) if source_bits else 0.0

    print()
    print("  " + "=" * 50)
    print("  │              评 估 指 标                   │")
    print("  " + "=" * 50)

    # 1. 准确率 (Accuracy)
    print(f"  │ 1. 准确率 (Accuracy)")
    print(f"  │    信道原始错误:       {ch_errors} 个编码 bit "
          f"({actual_error_rate*100:.2f}%)")
    print(f"  │    Viterbi 残留错误:   {final_errors} 个源 bit "
          f"({viterbi_ber*100:.3f}%)")
    if args.repeat > 1:
        print(f"  │    {args.repeat}x 重复+多数投票: 进一步消除残留错误")
    if actual_error_rate > 0 and viterbi_ber > 0:
        corrected = ch_errors - final_errors
        print(f"  │    Viterbi 净纠正:    约 {max(0, corrected)} 个错误")
    print(f"  │    源编码压缩率:       {img.size * 8 / len(source_bits):.1f}x "
          f"({len(source_bits)} bits)")

    # 2. 算法复杂度 (Algorithm Complexity)
    print(f"  │")
    print(f"  │ 2. 算法复杂度 (Algorithm Complexity)")
    print(f"  │    源编码:             {t_enc:.3f}s   (DCT+量化+RLE+Huffman)")
    print(f"  │    信道编码:           {t_ch_enc:.3f}s   (卷积码 码率1/2)")
    print(f"  │    交织:               {t_inter:.3f}s   (块交织 64×128)")
    print(f"  │    信道传输:           {t_tx:.3f}s")
    print(f"  │    解交织+信道译码:    {t_deinter + t_ch_dec:.3f}s   (Viterbi)")
    print(f"  │    源解码:             {t_dec:.3f}s   (Huffman+IDCT)")
    print(f"  │    ─────────────────────────────────")
    print(f"  │    端到端总耗时:       {t_total:.3f}s")
    print(f"  │    图像分辨率:         {img.shape[1]}×{img.shape[0]} "
          f"({img.shape[0] * img.shape[1] / 1000:.0f}k 像素)")

    # 3. 峰值信噪比 (PSNR)
    print(f"  │")
    print(f"  │ 3. 峰值信噪比 (PSNR)")
    if psnr == float('inf'):
        print(f"  │    PSNR = ∞ dB  (无损重建)")
    else:
        print(f"  │    PSNR = {psnr:.2f} dB")
        if psnr >= 30:
            q = "优秀 — 肉眼几乎不可察觉差异"
        elif psnr >= 25:
            q = "良好 — 轻微失真"
        elif psnr >= 20:
            q = "可接受 — 有可见噪声"
        elif psnr >= 15:
            q = "较差 — 明显失真"
        else:
            q = "差 — 严重损坏"
        print(f"  │    质量等级:           {q}")
    print(f"  │    输出文件:           {saved_path}")
    print("  " + "=" * 50)
    if args.save_bin:
        from src.bitstream import pack_bitstream
        bin_data = pack_bitstream(decoded_bits, header)
        os.makedirs(os.path.dirname(args.save_bin) or '.', exist_ok=True)
        with open(args.save_bin, 'wb') as f:
            f.write(bin_data)
        print(f"  比特流:      {args.save_bin}  ({len(bin_data)} bytes)")


if __name__ == '__main__':
    main()
