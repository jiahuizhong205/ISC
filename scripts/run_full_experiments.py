#!/usr/bin/env python
"""
ISC Project 2 — 完整实验脚本（修复版）

修复原版问题:
  1. 多随机种子 (3 seeds) 提供统计方差
  2. 增量保存 CSV，防止中途崩溃丢失数据
  3. 合理的参数空间避免 Viterbi Pure-Python 运行数小时
  4. 明确标注合成图像来源

用法:
    python scripts/run_full_experiments.py                    # 跑全部实验 (~20-30 min)
    python scripts/run_full_experiments.py --quick            # 快速验证模式 (~5 min)
    python scripts/run_full_experiments.py --csv results/new_analysis.csv
"""

import argparse
import csv
import os
import sys
import time
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.channel_model.channel import create_channel

# ═══════════════════════════════════════════════════════════════════════
# 参数空间
# ═══════════════════════════════════════════════════════════════════════

# 快速模式 vs 完整模式
BSC_PARAMS_QUICK = [0.0, 0.01, 0.05, 0.1]
BEC_PARAMS_QUICK = [0.0, 0.05, 0.1, 0.2]
BSC_PARAMS_FULL  = [0.0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1]
BEC_PARAMS_FULL  = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]

Q_VALUES_QUICK = [10, 50, 90]
Q_VALUES_FULL  = [5, 10, 25, 50, 75, 90]
SEEDS_QUICK    = [42]
SEEDS_FULL     = [42, 123, 456]
REPEAT_VALUES  = [1, 3, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ISC Project 2 — 完整实验 (修复版)')
    parser.add_argument('--quick', action='store_true',
                        help='快速验证模式 (更少参数, 约5分钟)')
    parser.add_argument('--medium', action='store_true',
                        help='中等模式: quick参数 + 3 seeds (推荐, ~15分钟)')
    parser.add_argument('--csv', type=str, default='results/analysis.csv',
                        help='CSV 输出路径')
    parser.add_argument('--no-repeats', action='store_true',
                        help='跳过 repeat 扫描 (大幅加速)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='逗号分隔的 seeds 列表, 如 "42,123,456"')
    parser.add_argument('--images', type=int, default=0,
                        help='使用前 N 张图片 (0=全部)')
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# 图像加载
# ═══════════════════════════════════════════════════════════════════════

def discover_kodak_images(subset: int = 0) -> Dict[str, np.ndarray]:
    """从 data/ 目录加载 PNG 图片，返回 {文件名: numpy_array}。"""
    from PIL import Image

    data_dir = 'data'
    if not os.path.isdir(data_dir):
        return {}

    images: Dict[str, np.ndarray] = {}
    files = sorted(f for f in os.listdir(data_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
    if subset and subset < len(files):
        files = files[:subset]

    for f in files:
        path = os.path.join(data_dir, f)
        try:
            img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
            images[f] = img
        except Exception as e:
            print(f"  [WARN] 跳过 {path}: {e}")

    return images


def generate_test_images(seed: int = 42) -> Dict[str, np.ndarray]:
    """生成 4 张 256×256 合成测试图像，覆盖不同频率特性（fallback）。"""
    rng = np.random.RandomState(seed)
    images: Dict[str, np.ndarray] = {}
    H, W = 256, 256

    # 1) 自然场景模拟: 平滑渐变 + 高频纹理
    x = np.linspace(0, 4 * np.pi, W)
    y = np.linspace(0, 4 * np.pi, H)
    xx, yy = np.meshgrid(x, y)
    r_ch = (np.sin(xx) * np.cos(yy) * 0.5 + 0.5) * 255
    g_ch = (np.sin(xx + 2) * np.sin(yy + 1) * 0.5 + 0.5) * 255
    b_ch = (np.cos(xx) * np.sin(yy + 0.5) * 0.5 + 0.5) * 255
    img_nature = np.stack([r_ch, g_ch, b_ch], axis=-1).astype(np.uint8)
    images['synthetic_nature'] = img_nature

    # 2) 几何图形: 边缘丰富
    img_geo = np.zeros((H, W, 3), dtype=np.uint8)
    img_geo[30:100, 40:200] = [220, 180, 50]
    img_geo[130:200, 60:120] = [50, 180, 220]
    img_geo[50:180, 150:220] = [100, 220, 80]
    rr, cc = np.ogrid[:H, :W]
    circle = (rr - 180) ** 2 + (cc - 180) ** 2 < 40 ** 2
    img_geo[circle] = [220, 50, 50]
    img_geo = img_geo.astype(np.float64)
    img_geo += np.linspace(0, 30, H)[:, None, None]
    img_geo = img_geo.clip(0, 255).astype(np.uint8)
    images['synthetic_geometric'] = img_geo

    # 3) 渐变: 平滑区域为主
    grad = np.linspace(0, 255, W).astype(np.uint8)
    img_grad = np.zeros((H, W, 3), dtype=np.uint8)
    img_grad[:, :, 0] = grad
    img_grad[:, :, 1] = grad[::-1]
    img_grad[:, :, 2] = np.linspace(0, 255, H)[:, None].astype(np.uint8)
    images['synthetic_gradient'] = img_grad

    # 4) 噪声纹理: 全高频
    img_noise = rng.randint(0, 256, (H, W, 3), dtype=np.uint8)
    images['synthetic_noise'] = img_noise

    return images


# ═══════════════════════════════════════════════════════════════════════
# 评估指标
# ═══════════════════════════════════════════════════════════════════════

def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = np.mean(diff ** 2)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray,
                 K1: float = 0.01, K2: float = 0.03,
                 win_size: int = 11, sigma: float = 1.5) -> float:
    """结构相似性指数 (SSIM)。"""
    L = 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    half = win_size // 2
    x_coords = np.arange(-half, half + 1, dtype=np.float64)
    gauss_1d = np.exp(-(x_coords ** 2) / (2 * sigma ** 2))
    gauss_1d /= gauss_1d.sum()

    if original.ndim == 3:
        ssim_ch = []
        for c in range(original.shape[2]):
            ssim_ch.append(_ssim_channel(
                original[:, :, c].astype(np.float64),
                reconstructed[:, :, c].astype(np.float64),
                gauss_1d, C1, C2))
        return float(np.mean(ssim_ch))
    else:
        return _ssim_channel(original.astype(np.float64),
                             reconstructed.astype(np.float64),
                             gauss_1d, C1, C2)


def _ssim_channel(img1: np.ndarray, img2: np.ndarray,
                   gauss_1d: np.ndarray, C1: float, C2: float) -> float:
    mu1 = _separable_conv2d(img1, gauss_1d)
    mu2 = _separable_conv2d(img2, gauss_1d)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _separable_conv2d(img1 * img1, gauss_1d) - mu1_sq
    sigma2_sq = _separable_conv2d(img2 * img2, gauss_1d) - mu2_sq
    sigma12 = _separable_conv2d(img1 * img2, gauss_1d) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def _separable_conv2d(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    tmp = np.apply_along_axis(
        lambda r: np.convolve(r, kernel_1d, mode='same'), 1, img)
    result = np.apply_along_axis(
        lambda c: np.convolve(c, kernel_1d, mode='same'), 0, tmp)
    return result


def compute_ber(original_bits: List[int], decoded_bits: List[int]) -> float:
    if not original_bits:
        return 0.0
    errors = sum(1 for a, b in zip(original_bits, decoded_bits) if a != b)
    return errors / len(original_bits)


# ═══════════════════════════════════════════════════════════════════════
# 单次实验
# ═══════════════════════════════════════════════════════════════════════

def run_single(image_name: str, original: np.ndarray,
               channel_type: str, param: float,
               quality: int, repeat: int, seed: int) -> Dict[str, Any]:
    """执行单次端到端实验。"""
    result: Dict[str, Any] = {
        'image': image_name,
        'channel': channel_type,
        'param': param,
        'quality': quality,
        'repeat': repeat,
        'seed': seed,
        'psnr': None,
        'ssim': None,
        'compression_ratio': None,
        'time_source_enc': None,
        'time_channel_enc': None,
        'time_interleave': None,
        'time_transmission': None,
        'time_channel_dec': None,
        'time_source_dec': None,
        'actual_error_rate': None,
        'viterbi_ber': None,
        'source_bits': None,
        'encoded_bits': None,
    }

    # ── 源编码 ──
    header: Dict[str, Any] = {}
    from src.source_coding.encoder import DCTEncoder
    t0 = time.perf_counter()
    encoder = DCTEncoder(quality=quality, repeat=repeat)
    encoded = encoder.encode(original)
    source_bits = encoded['bits']
    header = encoded['header']
    result['time_source_enc'] = time.perf_counter() - t0
    result['source_bits'] = len(source_bits)
    result['compression_ratio'] = (
        original.size * 8 / len(source_bits) if source_bits else float('inf')
    )

    # ── 信道编码 ──
    from src.channel_coding.convolutional import ConvCodec
    t0 = time.perf_counter()
    channel_codec = ConvCodec()
    encoded_bits = channel_codec.encode(source_bits)
    result['time_channel_enc'] = time.perf_counter() - t0
    result['encoded_bits'] = len(encoded_bits)

    # ── 交织 ──
    from src.channel_coding.interleaver import BlockInterleaver
    interleaver = None
    encoded_len_before_interleave = len(encoded_bits)
    t0 = time.perf_counter()
    interleaver = BlockInterleaver(rows=64, cols=128)
    tx_bits = interleaver.interleave(encoded_bits)
    result['time_interleave'] = time.perf_counter() - t0

    # ── 信道传输 ──
    channel = create_channel(channel_type, param, seed=seed)
    t0 = time.perf_counter()
    received, actual_rate = channel.transmit(tx_bits)
    result['time_transmission'] = time.perf_counter() - t0
    result['actual_error_rate'] = actual_rate

    # ── 解交织 + 信道译码 ──
    from src.channel_coding.convolutional import ConvCodec
    t0 = time.perf_counter()
    if interleaver is not None:
        deinterleaved = interleaver.deinterleave(received)
        deinterleaved = deinterleaved[:encoded_len_before_interleave]
    else:
        deinterleaved = list(received)
    t_deinter = time.perf_counter() - t0

    t0 = time.perf_counter()
    channel_codec2 = ConvCodec()
    decoded_bits = channel_codec2.decode(deinterleaved, channel_type=channel_type)
    t_viterbi = time.perf_counter() - t0
    result['time_channel_dec'] = t_viterbi + t_deinter

    # ── Viterbi BER ──
    result['viterbi_ber'] = compute_ber(source_bits, decoded_bits)

    # ── 源解码 ──
    from src.source_coding.decoder import DCTDecoder
    t0 = time.perf_counter()
    decoder = DCTDecoder()
    recovered = decoder.decode(decoded_bits, header)
    result['time_source_dec'] = time.perf_counter() - t0

    # ── PSNR & SSIM ──
    if recovered.shape != original.shape:
        recovered = _resize_to_match(recovered, original.shape)
    result['psnr'] = compute_psnr(original, recovered)
    result['ssim'] = compute_ssim(original, recovered)

    return result


def _resize_to_match(img: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    if img.shape == target_shape:
        return img
    result = np.zeros(target_shape, dtype=img.dtype)
    h = min(img.shape[0], target_shape[0])
    w = min(img.shape[1], target_shape[1])
    result[:h, :w, :min(img.shape[2], target_shape[2])] = \
        img[:h, :w, :min(img.shape[2], target_shape[2])]
    return result


# ═══════════════════════════════════════════════════════════════════════
# CSV 工具
# ═══════════════════════════════════════════════════════════════════════

CSV_FIELDNAMES = [
    'image', 'channel', 'param', 'quality', 'repeat', 'seed',
    'psnr', 'ssim', 'compression_ratio',
    'time_source_enc', 'time_channel_enc', 'time_interleave',
    'time_transmission', 'time_channel_dec', 'time_source_dec',
    'actual_error_rate', 'viterbi_ber',
    'source_bits', 'encoded_bits',
]


def init_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()


def append_csv(path: str, row: Dict[str, Any]) -> None:
    """增量追加单行到 CSV。"""
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # 模式选择: medium(推荐) > quick > full
    if args.medium:
        bsc_params = BSC_PARAMS_QUICK
        bec_params = BEC_PARAMS_QUICK
        q_values = Q_VALUES_QUICK
        seeds = [42]  # 12张Kodak图 × 1seed ≈ 2h
    elif args.quick:
        bsc_params = BSC_PARAMS_QUICK
        bec_params = BEC_PARAMS_QUICK
        q_values = Q_VALUES_QUICK
        seeds = [42]
    else:
        bsc_params = BSC_PARAMS_FULL
        bec_params = BEC_PARAMS_FULL
        q_values = Q_VALUES_FULL
        seeds = SEEDS_FULL

    # 命令行指定 seeds 覆盖
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # 从 data/ 加载 Kodak 真实图片
    images = discover_kodak_images(subset=args.images)
    if not images:
        print("[ERROR] data/ 目录未找到图片，请放入 PNG 文件")
        sys.exit(1)
    print(f"使用 {len(images)} 张 Kodak 真实图片:")
    for name, img in images.items():
        print(f"  - {name}: {img.shape[1]}×{img.shape[0]}, {img.shape[2]} channels")

    # 构建实验列表
    experiments: List[Tuple] = []
    # 主要实验: 遍历信道 × 参数 × Q × 图像 × seed（repeat=1）
    for ch_type in ['bsc']:
        for param in bsc_params:
            for q in q_values:
                for (img_name, img) in images.items():
                    for seed in seeds:
                        experiments.append((ch_type, param, q, img_name, img, 1, seed))
    for ch_type in ['bec']:
        for param in bec_params:
            for q in q_values:
                for (img_name, img) in images.items():
                    for seed in seeds:
                        experiments.append((ch_type, param, q, img_name, img, 1, seed))

    # Repeat 扫描 (仅用第一张图 + 中等参数)
    if not args.no_repeats:
        first_img_name = list(images.keys())[0]
        first_img = images[first_img_name]
        for seed in seeds[:1]:  # 仅单种子
            for repeat in [3, 5]:
                experiments.append(('bsc', 0.05, 50, first_img_name, first_img, repeat, seed))
                experiments.append(('bec', 0.1, 50, first_img_name, first_img, repeat, seed))

    total = len(experiments)
    avg_img_pixels = int(np.mean([img.size for img in images.values()]))
    est_sec_per_exp = avg_img_pixels / (256 * 256 * 3) * 1.5  # 相对于256×256的缩放
    print(f"\n参数空间:")
    print(f"  BSC ε ∈ {bsc_params}")
    print(f"  BEC p ∈ {bec_params}")
    print(f"  Q     ∈ {q_values}")
    print(f"  Seeds ∈ {seeds}")
    print(f"  Repeat扫描: {'跳过' if args.no_repeats else '包含 (仅第一张图)'}")
    print(f"  平均图像尺寸: {avg_img_pixels/1000:.0f}k 像素")
    print(f"  总计 {total} 组实验")
    print(f"  预计耗时: ~{total * est_sec_per_exp / 60:.0f} 分钟 (Viterbi 纯 Python)")
    print("=" * 60)

    # 初始化 CSV
    init_csv(args.csv)

    count = 0
    t_start = time.time()

    for ch_type, param, q, img_name, img, repeat, seed in experiments:
        count += 1
        t_exp_start = time.time()

        try:
            result = run_single(img_name, img, ch_type, param, q, repeat, seed)
            elapsed = time.time() - t_exp_start
            append_csv(args.csv, result)

            psnr_str = (f"{result['psnr']:.1f}" if result['psnr'] != float('inf')
                        else "∞")
            print(f"  [{count:4d}/{total}] {img_name:25s} "
                  f"ch={ch_type.upper()} p={param:.3f} Q={q:2d} "
                  f"R={repeat} seed={seed:3d}  "
                  f"PSNR={psnr_str:>6s} dB  SSIM={result['ssim']:.4f}  "
                  f"CR={result['compression_ratio']:.1f}x  "
                  f"Viterbi={result['viterbi_ber']*100:.3f}%%  "
                  f"t={elapsed:.1f}s")
        except Exception as e:
            print(f"  [{count:4d}/{total}] {img_name:25s} "
                  f"ch={ch_type.upper()} p={param:.3f} Q={q:2d} "
                  f"R={repeat} seed={seed:3d}  "
                  f"[FAIL] {e}")
            import traceback
            traceback.print_exc()

        # 每 20 组打印进度预估
        if count % 20 == 0:
            elapsed_total = time.time() - t_start
            eta = elapsed_total / count * (total - count)
            print(f"  --- 进度: {count}/{total} ({100*count/total:.0f}%%), "
                  f"已用时 {elapsed_total/60:.1f}min, "
                  f"预计剩余 {eta/60:.1f}min ---")

    # 汇总
    t_total = time.time() - t_start
    print()
    print("=" * 60)
    print(f"  实验完成!")
    print(f"  总耗时: {t_total/60:.1f} 分钟 ({t_total:.0f} 秒)")
    print(f"  实验数: {total}")
    print(f"  结果保存: {args.csv}")
    print("=" * 60)


if __name__ == '__main__':
    main()
