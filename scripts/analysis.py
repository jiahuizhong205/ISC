#!/usr/bin/env python
"""
ISC Project 2 — 综合性能分析与可视化

分析维度:
  1. PSNR / SSIM vs 信道错误率 (BSC ε, BEC p)
  2. PSNR / SSIM vs 源编码质量因子 Q
  3. 压缩率 vs 质量因子
  4. 算法复杂度 (各阶段耗时)
  5. 率失真曲线 (PSNR vs Compression Ratio)
  6. Viterbi 纠错性能 (信道BER vs 残留BER)

用法:
    python scripts/analysis.py                        # 完整分析
    python scripts/analysis.py --quick                # 快速模式 (少量参数)
    python scripts/analysis.py --no-charts            # 仅数据，不绘图
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════

BSC_PARAMS_FULL = [0.0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15]
BEC_PARAMS_FULL = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
BSC_PARAMS_QUICK = [0.0, 0.01, 0.05, 0.1]
BEC_PARAMS_QUICK = [0.0, 0.05, 0.1, 0.2]
Q_VALUES_FULL = [5, 10, 25, 50, 75, 90, 95]
Q_VALUES_QUICK = [10, 50, 90]
REPEAT_VALUES = [1, 3, 5]
DEFAULT_SEED = 42
OUTPUT_DIR = 'results/figures'
CSV_PATH = 'results/analysis.csv'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ISC Project 2 — 性能分析')
    parser.add_argument('--quick', action='store_true', help='快速模式 (减少参数组合)')
    parser.add_argument('--no-charts', action='store_true', help='跳过图表生成')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--csv', type=str, default=CSV_PATH)
    parser.add_argument('--images', type=int, default=0, help='使用前 N 张 kodak 图像 (0=生成合成图)')
    parser.add_argument('--from-csv', type=str, default=None, help='从已有 CSV 加载数据并重新绘图 (跳过实验)')
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# 测试图像生成 (若无 kodak 图集则合成)
# ═══════════════════════════════════════════════════════════════════════

def generate_test_images(seed: int = 42) -> Dict[str, np.ndarray]:
    """生成 4 张合成测试图像，覆盖不同频率特性。"""
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
    img_geo[30:100, 40:200] = [220, 180, 50]       # 横条
    img_geo[130:200, 60:120] = [50, 180, 220]       # 竖条
    img_geo[50:180, 150:220] = [100, 220, 80]       # 大方块
    rr, cc = np.ogrid[:H, :W]
    circle = (rr - 180) ** 2 + (cc - 180) ** 2 < 40 ** 2
    img_geo[circle] = [220, 50, 50]                  # 红色圆形
    # 添加渐变背景
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


def discover_kodak_images(subset: int = 0) -> Dict[str, np.ndarray]:
    """尝试加载 kodak 图集。"""
    from PIL import Image

    search_dirs = ['data/kodak', 'data']
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
                img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
                images[f] = img
            except Exception as e:
                print(f"  [WARN] 跳过 {path}: {e}")
        if images:
            break
    return images


# ═══════════════════════════════════════════════════════════════════════
# 评估指标
# ═══════════════════════════════════════════════════════════════════════

def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """峰值信噪比 (dB)。"""
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = np.mean(diff ** 2)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray,
                 K1: float = 0.01, K2: float = 0.03,
                 win_size: int = 11, sigma: float = 1.5) -> float:
    """
    结构相似性指数 (SSIM)。

    使用高斯加权窗口计算局部 SSIM，返回整图均值。

    SSIM(x, y) = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))

    参考文献: Wang et al., "Image Quality Assessment: From Error Visibility
              to Structural Similarity", IEEE TIP, 2004.
    """
    L = 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # 1-D 高斯窗口
    half = win_size // 2
    x_coords = np.arange(-half, half + 1, dtype=np.float64)
    gauss_1d = np.exp(-(x_coords ** 2) / (2 * sigma ** 2))
    gauss_1d /= gauss_1d.sum()

    # 可分离滤波加速
    if original.ndim == 3:
        # 逐通道计算后取均值
        ssim_ch = []
        for c in range(original.shape[2]):
            ssim_ch.append(_ssim_channel(
                original[:, :, c].astype(np.float64),
                reconstructed[:, :, c].astype(np.float64),
                gauss_1d, C1, C2))
        return float(np.mean(ssim_ch))
    else:
        return _ssim_channel(
            original.astype(np.float64),
            reconstructed.astype(np.float64),
            gauss_1d, C1, C2)


def _ssim_channel(img1: np.ndarray, img2: np.ndarray,
                   gauss_1d: np.ndarray, C1: float, C2: float) -> float:
    """单通道 SSIM 计算。"""
    # 高斯平滑
    mu1 = _separable_conv2d(img1, gauss_1d)
    mu2 = _separable_conv2d(img2, gauss_1d)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _separable_conv2d(img1 * img1, gauss_1d) - mu1_sq
    sigma2_sq = _separable_conv2d(img2 * img2, gauss_1d) - mu2_sq
    sigma12 = _separable_conv2d(img1 * img2, gauss_1d) - mu1_mu2

    # SSIM 图
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


def _separable_conv2d(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """可分离 2-D 卷积 (mode='same')。"""
    # 水平
    tmp = np.apply_along_axis(
        lambda r: np.convolve(r, kernel_1d, mode='same'), 1, img)
    # 垂直
    result = np.apply_along_axis(
        lambda c: np.convolve(c, kernel_1d, mode='same'), 0, tmp)
    return result


def compute_ber(original_bits: List[int], decoded_bits: List[int]) -> float:
    """比特错误率 (BER)。"""
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
    """执行单次端到端实验，返回完整指标字典。"""
    result: Dict[str, Any] = {
        'image': image_name,
        'channel': channel_type,
        'param': param,
        'quality': quality,
        'repeat': repeat,
        'psnr': None,
        'ssim': None,
        'time_source_enc': None,
        'time_channel_enc': None,
        'time_interleave': None,
        'time_transmission': None,
        'time_channel_dec': None,
        'time_source_dec': None,
        'compression_ratio': None,
        'actual_error_rate': None,
        'viterbi_ber': None,
        'source_bits': None,
        'encoded_bits': None,
    }

    H, W = original.shape[:2]

    # ── 源编码 ──
    header: Dict[str, Any] = {}
    try:
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
    except ImportError:
        source_bits = _image_to_raw_bits(original)
        header = {'shape': list(original.shape), 'fallback': True}
        result['time_source_enc'] = 0.0
        result['compression_ratio'] = 1.0
        result['source_bits'] = len(source_bits)

    # ── 信道编码 ──
    channel_codec = None
    try:
        from src.channel_coding.convolutional import ConvCodec
        t0 = time.perf_counter()
        channel_codec = ConvCodec()
        encoded_bits = channel_codec.encode(source_bits)
        result['time_channel_enc'] = time.perf_counter() - t0
    except ImportError:
        encoded_bits = list(source_bits)
        result['time_channel_enc'] = 0.0
    result['encoded_bits'] = len(encoded_bits)

    # ── 交织 ──
    interleaver = None
    encoded_len_before_interleave = len(encoded_bits)
    try:
        from src.channel_coding.interleaver import BlockInterleaver
        t0 = time.perf_counter()
        interleaver = BlockInterleaver(rows=64, cols=128)
        tx_bits = interleaver.interleave(encoded_bits)
        result['time_interleave'] = time.perf_counter() - t0
    except ImportError:
        tx_bits = list(encoded_bits)
        result['time_interleave'] = 0.0

    # ── 信道传输 ──
    from src.channel_model.channel import create_channel
    channel = create_channel(channel_type, param, seed=seed)
    t0 = time.perf_counter()
    received, actual_rate = channel.transmit(tx_bits)
    result['time_transmission'] = time.perf_counter() - t0
    result['actual_error_rate'] = actual_rate

    # ── 解交织 + 信道译码 ──
    try:
        from src.channel_coding.convolutional import ConvCodec
        t0 = time.perf_counter()
        if interleaver is not None:
            deinterleaved = interleaver.deinterleave(received)
            deinterleaved = deinterleaved[:encoded_len_before_interleave]
        else:
            deinterleaved = list(received)
        t_deinter = time.perf_counter() - t0

        t0 = time.perf_counter()
        if channel_codec is None:
            channel_codec = ConvCodec()
        decoded_bits = channel_codec.decode(deinterleaved, channel_type=channel_type)
        result['time_channel_dec'] = time.perf_counter() - t0 + t_deinter
    except ImportError:
        if interleaver is not None:
            received = interleaver.deinterleave(received)[:encoded_len_before_interleave]
        decoded_bits = [int(b) if b is not None else 0 for b in received]
        result['time_channel_dec'] = 0.0

    # ── Viterbi BER ──
    result['viterbi_ber'] = compute_ber(source_bits, decoded_bits)

    # ── 源解码 ──
    try:
        from src.source_coding.decoder import DCTDecoder
        t0 = time.perf_counter()
        decoder = DCTDecoder()
        recovered = decoder.decode(decoded_bits, header)
        result['time_source_dec'] = time.perf_counter() - t0
    except ImportError:
        recovered = original
        result['time_source_dec'] = 0.0

    # ── PSNR & SSIM ──
    if recovered.shape != original.shape:
        recovered = _resize_to_match(recovered, original.shape)

    result['psnr'] = compute_psnr(original, recovered)
    result['ssim'] = compute_ssim(original, recovered)

    return result


def _image_to_raw_bits(img: np.ndarray) -> List[int]:
    """像素展平为比特流（降级模式）。"""
    bits: List[int] = []
    for val in img.flatten():
        for shift in range(7, -1, -1):
            bits.append((int(val) >> shift) & 1)
    return bits


def _resize_to_match(img: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """裁剪/填充使形状匹配。"""
    if img.shape == target_shape:
        return img
    result = np.zeros(target_shape, dtype=img.dtype)
    h = min(img.shape[0], target_shape[0])
    w = min(img.shape[1], target_shape[1])
    result[:h, :w, :min(img.shape[2], target_shape[2])] = img[:h, :w, :min(img.shape[2], target_shape[2])]
    return result


# ═══════════════════════════════════════════════════════════════════════
# 主分析流程
# ═══════════════════════════════════════════════════════════════════════

def run_analysis(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """执行完整分析，返回所有实验结果。"""
    # 加载图像
    images = discover_kodak_images(subset=args.images)
    if not images:
        print("[INFO] 未找到 kodak 图集，使用合成测试图像")
        images = generate_test_images(seed=args.seed)

    print(f"使用 {len(images)} 张测试图像:")
    for name, img in images.items():
        print(f"  - {name}: {img.shape}")

    # 参数空间
    bsc_params = BSC_PARAMS_QUICK if args.quick else BSC_PARAMS_FULL
    bec_params = BEC_PARAMS_QUICK if args.quick else BEC_PARAMS_FULL
    q_values = Q_VALUES_QUICK if args.quick else Q_VALUES_FULL

    # 构建实验列表
    experiments_bsc = list(product(
        ['bsc'], bsc_params, q_values, images.items(), REPEAT_VALUES[:1]))
    experiments_bec = list(product(
        ['bec'], bec_params, q_values, images.items(), REPEAT_VALUES[:1]))
    # 独立研究 repeat 的影响 (用中等参数)
    repeat_exps = list(product(
        ['bsc'], [0.05], [50], list(images.items())[:1], REPEAT_VALUES))
    repeat_exps += list(product(
        ['bec'], [0.1], [50], list(images.items())[:1], REPEAT_VALUES))

    all_experiments = experiments_bsc + experiments_bec + repeat_exps
    total = len(all_experiments)

    print(f"\n参数空间:")
    print(f"  BSC ε ∈ {bsc_params}")
    print(f"  BEC p ∈ {bec_params}")
    print(f"  Q     ∈ {q_values}")
    print(f"  总计 {total} 组实验")
    print("=" * 60)

    results: List[Dict[str, Any]] = []
    count = 0

    for ch_type, param, q, (img_name, img), repeat in all_experiments:
        count += 1
        result = run_single(
            img_name, img, ch_type, param, q, repeat, args.seed)

        # 跳过 repeat 重复记录 (仅记录变化的部分)
        if repeat == REPEAT_VALUES[0]:
            result['_keep'] = True
        else:
            result['_keep'] = False

        results.append(result)

        psnr_str = f"{result['psnr']:.1f}" if result['psnr'] != float('inf') else "∞"
        print(f"  [{count:4d}/{total}] {img_name:25s} "
              f"{ch_type.upper()} p={param:.3f} Q={q:2d} "
              f"PSNR={psnr_str:>6s} dB  SSIM={result['ssim']:.4f}  "
              f"CR={result['compression_ratio']:.1f}x  "
              f"t={result['time_source_enc'] + result['time_channel_enc'] + result['time_channel_dec'] + result['time_source_dec']:.3f}s")

    return results


# ═══════════════════════════════════════════════════════════════════════
# 统计聚合
# ═══════════════════════════════════════════════════════════════════════

def aggregate_by(results: List[Dict], group_key: str) -> Dict[Any, List[Dict]]:
    """按指定键分组。"""
    groups: Dict[Any, List[Dict]] = defaultdict(list)
    for r in results:
        groups[r[group_key]].append(r)
    return dict(groups)


def mean_std(values: List[float]) -> Tuple[float, float]:
    """计算均值和标准差。"""
    finite = [v for v in values if v is not None and v != float('inf')]
    if not finite:
        return 0.0, 0.0
    return float(np.mean(finite)), float(np.std(finite))


# ═══════════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════════

def setup_plot_style():
    """配置 matplotlib 中英文混排样式。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 9,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    return plt


def save_figure(fig, name: str):
    """保存图表到 results/figures/。"""
    import matplotlib.pyplot as plt
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  [OK] {path}")


def plot_psnr_vs_error_rate(results: List[Dict], plt_module):
    """
    图 1: PSNR vs 信道错误率
    - 左: BSC, 右: BEC
    - 每条线代表一个 quality factor
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(Q_VALUES_FULL)))
    q_color = {q: colors[i] for i, q in enumerate(Q_VALUES_FULL)}

    markers = ['o', 's', 'D', '^', 'v', 'p', '*']

    for ax_idx, (ch_type, label) in enumerate([('bsc', 'BSC'), ('bec', 'BEC')]):
        ax = axes[ax_idx]
        ch_results = [r for r in results if r['channel'] == ch_type and r.get('_keep', True)]

        for qi, q in enumerate(Q_VALUES_FULL if len(set(r['quality'] for r in ch_results)) > 3 else Q_VALUES_QUICK):
            q_results = [r for r in ch_results if r['quality'] == q]
            if not q_results:
                continue

            by_param = aggregate_by(q_results, 'param')
            params = sorted(by_param.keys())
            psnr_means = [mean_std([r['psnr'] for r in by_param[p] if r['psnr'] != float('inf')])[0] for p in params]
            psnr_stds = [mean_std([r['psnr'] for r in by_param[p] if r['psnr'] != float('inf')])[1] for p in params]

            marker = markers[qi % len(markers)]
            color = q_color.get(q, colors[qi % len(colors)])
            ax.errorbar(params, psnr_means, yerr=psnr_stds,
                        marker=marker, color=color, linewidth=1.8, markersize=6,
                        capsize=3, label=f'Q={q}')

        ax.set_xlabel(f'{label} Error Probability')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{label} Channel: PSNR vs Error Rate')
        ax.legend(loc='lower left', ncol=2, framealpha=0.8)
        ax.set_ylim(bottom=0)

    fig.suptitle('PSNR vs Channel Error Rate (by Quality Factor)', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, '01_psnr_vs_error_rate.png')


def plot_ssim_vs_error_rate(results: List[Dict], plt_module):
    """
    图 2: SSIM vs 信道错误率
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(Q_VALUES_FULL)))
    markers = ['o', 's', 'D', '^', 'v', 'p', '*']

    for ax_idx, (ch_type, label) in enumerate([('bsc', 'BSC'), ('bec', 'BEC')]):
        ax = axes[ax_idx]
        ch_results = [r for r in results if r['channel'] == ch_type and r.get('_keep', True)]
        unique_qs = sorted(set(r['quality'] for r in ch_results))

        for qi, q in enumerate(unique_qs):
            q_results = [r for r in ch_results if r['quality'] == q]
            if not q_results:
                continue

            by_param = aggregate_by(q_results, 'param')
            params = sorted(by_param.keys())
            ssim_means = [mean_std([r['ssim'] for r in by_param[p] if r['ssim'] is not None])[0] for p in params]
            ssim_stds = [mean_std([r['ssim'] for r in by_param[p] if r['ssim'] is not None])[1] for p in params]

            marker = markers[qi % len(markers)]
            color = colors[qi % len(colors)]
            ax.errorbar(params, ssim_means, yerr=ssim_stds,
                        marker=marker, color=color, linewidth=1.8, markersize=6,
                        capsize=3, label=f'Q={q}')

        ax.set_xlabel(f'{label} Error Probability')
        ax.set_ylabel('SSIM')
        ax.set_title(f'{label} Channel: SSIM vs Error Rate')
        ax.legend(loc='lower left', ncol=2, framealpha=0.8)
        ax.set_ylim(0, 1.05)

    fig.suptitle('SSIM vs Channel Error Rate (by Quality Factor)', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, '02_ssim_vs_error_rate.png')


def plot_compression_ratio(results: List[Dict], plt_module):
    """
    图 3: 压缩率 vs 质量因子
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    main_results = [r for r in results if r.get('_keep', True)]
    by_q = aggregate_by(main_results, 'quality')
    qualities = sorted(by_q.keys())

    # 左: 压缩率
    ax = axes[0]
    cr_means = [mean_std([r['compression_ratio'] for r in by_q[q]
                          if r['compression_ratio'] and r['compression_ratio'] != float('inf')])[0]
                for q in qualities]
    cr_stds = [mean_std([r['compression_ratio'] for r in by_q[q]
                         if r['compression_ratio'] and r['compression_ratio'] != float('inf')])[1]
               for q in qualities]

    ax.bar(qualities, cr_means, width=2.5, color=plt.cm.Blues(0.6),
           edgecolor='white', linewidth=0.5)
    ax.errorbar(qualities, cr_means, yerr=cr_stds, fmt='none',
                ecolor='#333333', capsize=4, linewidth=1)
    ax.set_xlabel('Quality Factor Q')
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Compression Ratio vs Quality Factor')
    for i, (q, cr) in enumerate(zip(qualities, cr_means)):
        ax.text(q, cr + 0.3, f'{cr:.1f}x', ha='center', fontsize=8, color='#333')

    # 右: 源编码 bit 数
    ax = axes[1]
    bit_means = [mean_std([r['source_bits'] for r in by_q[q] if r['source_bits']])[0] / 1000
                 for q in qualities]
    ax.bar(qualities, bit_means, width=2.5, color=plt.cm.Oranges(0.6),
           edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Quality Factor Q')
    ax.set_ylabel('Source Bits (kbits)')
    ax.set_title('Compressed Size vs Quality Factor')
    for q, b in zip(qualities, bit_means):
        ax.text(q, b + max(bit_means) * 0.02, f'{b:.0f}k', ha='center', fontsize=8, color='#333')

    fig.suptitle('Source Coding Compression Performance', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, '03_compression_ratio.png')


def plot_complexity_breakdown(results: List[Dict], plt_module):
    """
    图 4: 算法复杂度分解
    - 左: 堆叠柱状图 (各阶段耗时)
    - 右: 饼图 (平均耗时占比)
    """
    import matplotlib.pyplot as plt

    main_results = [r for r in results if r.get('_keep', True)]
    by_q = aggregate_by(main_results, 'quality')
    qualities = sorted(by_q.keys())

    stages = ['time_source_enc', 'time_channel_enc', 'time_interleave',
              'time_transmission', 'time_channel_dec', 'time_source_dec']
    stage_labels = ['Source Enc', 'Channel Enc', 'Interleave',
                    'Transmission', 'Channel Dec', 'Source Dec']
    stage_colors = plt.cm.Set2(np.linspace(0, 1, len(stages)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左: 堆叠柱状图
    ax = axes[0]
    x = np.arange(len(qualities))
    width = 0.6
    bottoms = np.zeros(len(qualities))

    for si, (stage, label, color) in enumerate(zip(stages, stage_labels, stage_colors)):
        means = [mean_std([r[stage] for r in by_q[q] if r[stage] is not None])[0] for q in qualities]
        ax.bar(x, means, width, bottom=bottoms, label=label, color=color,
               edgecolor='white', linewidth=0.3)
        bottoms += np.array(means)

    ax.set_xlabel('Quality Factor Q')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Pipeline Time Breakdown by Stage')
    ax.set_xticks(x)
    ax.set_xticklabels(qualities)
    ax.legend(loc='upper left', framealpha=0.8, fontsize=8)

    # 右: 饼图 (整体平均)
    ax = axes[1]
    stage_totals = []
    for stage in stages:
        total = np.mean([r[stage] for r in main_results if r[stage] is not None and r[stage] > 0])
        stage_totals.append(max(total, 0))

    # 过滤零值
    non_zero = [(l, t) for l, t in zip(stage_labels, stage_totals) if t > 0]
    if non_zero:
        labels, values = zip(*non_zero)
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=stage_colors[:len(labels)],
            startangle=90, pctdistance=0.6)
        for at in autotexts:
            at.set_fontsize(8)
        ax.set_title('Average Time Distribution')

    fig.suptitle('Algorithm Complexity Analysis', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, '04_complexity_breakdown.png')


def plot_rate_distortion(results: List[Dict], plt_module):
    """
    图 5: 率失真曲线 (PSNR vs Bits Per Pixel)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))

    main_results = [r for r in results if r.get('_keep', True)]

    # 按 quality 分组
    by_q = aggregate_by(main_results, 'quality')
    qualities = sorted(by_q.keys())
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(qualities)))

    for qi, q in enumerate(qualities):
        q_results = by_q[q]
        # BPP = source_bits / (H * W * 3)
        bpps = []
        psnrs = []
        for r in q_results:
            if r['source_bits']:
                img_pixels = sum(1 for _ in filter(None, [r.get('_img_h'), r.get('_img_w')])) or 256 * 256 * 3
                # Estimate from any image
                bpps.append(r['source_bits'] / (256 * 256 * 3) if 'synthetic' in r['image'] else r['source_bits'] / (256 * 256 * 3))
                if r['psnr'] != float('inf'):
                    psnrs.append(r['psnr'])

        if bpps and psnrs:
            ax.scatter(bpps, psnrs, c=[colors[qi]], s=50, alpha=0.7,
                       edgecolors='white', linewidth=0.5, label=f'Q={q}')

    ax.set_xlabel('Bits Per Pixel (bpp)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Rate-Distortion: PSNR vs Bits Per Pixel')
    ax.legend(loc='lower right', framealpha=0.8)
    ax.set_xlim(left=0)

    fig.tight_layout()
    save_figure(fig, '05_rate_distortion.png')


def plot_viterbi_performance(results: List[Dict], plt_module):
    """
    图 6: Viterbi 纠错性能
    - 信道 BER vs 残留 BER
    """
    import matplotlib.pyplot as plt

    main_results = [r for r in results if r.get('_keep', True)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, (ch_type, label) in enumerate([('bsc', 'BSC'), ('bec', 'BEC')]):
        ax = axes[ax_idx]
        ch_results = [r for r in main_results if r['channel'] == ch_type]

        # 分组
        by_param = aggregate_by(ch_results, 'param')
        params = sorted(by_param.keys())

        ch_bers = []
        viterbi_bers = []
        for p in params:
            ch_bers.append(mean_std([r['actual_error_rate'] for r in by_param[p]
                                     if r['actual_error_rate'] is not None])[0])
            viterbi_bers.append(mean_std([r['viterbi_ber'] for r in by_param[p]
                                          if r['viterbi_ber'] is not None])[0])

        ax.plot(params, ch_bers, 'o-', color='#e74c3c', linewidth=2, markersize=7,
                label='Channel BER (before Viterbi)')
        ax.plot(params, viterbi_bers, 's-', color='#2ecc71', linewidth=2, markersize=7,
                label='Residual BER (after Viterbi)')

        # 标注改善倍数
        for p, ch, vi in zip(params, ch_bers, viterbi_bers):
            if ch > 0 and vi > 0:
                improvement = ch / vi
                ax.annotate(f'{improvement:.0f}x', (p, vi),
                            textcoords="offset points", xytext=(0, 12),
                            fontsize=8, ha='center', color='#2ecc71')

        ax.set_xlabel(f'{label} Error Probability')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.set_title(f'{label}: Viterbi Error Correction')
        ax.legend(framealpha=0.8)
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-6)

    fig.suptitle('Viterbi Decoder Performance', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, '06_viterbi_performance.png')


def plot_psnr_vs_quality(results: List[Dict], plt_module):
    """
    图 7: PSNR vs Quality Factor (在不同信道条件下)
    """
    import matplotlib.pyplot as plt

    main_results = [r for r in results if r.get('_keep', True)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, 8))

    for ax_idx, (ch_type, label) in enumerate([('bsc', 'BSC'), ('bec', 'BEC')]):
        ax = axes[ax_idx]
        ch_results = [r for r in main_results if r['channel'] == ch_type]

        unique_params = sorted(set(r['param'] for r in ch_results))
        for pi, p in enumerate(unique_params[:6]):  # 最多6条线
            p_results = [r for r in ch_results if r['param'] == p]
            by_q = aggregate_by(p_results, 'quality')
            qs = sorted(by_q.keys())
            psnrs = []
            for q in qs:
                finite = [r['psnr'] for r in by_q[q] if r['psnr'] != float('inf')]
                psnrs.append(np.mean(finite) if finite else 0)

            color = colors[pi]
            ax.plot(qs, psnrs, marker='o', color=color, linewidth=1.8, markersize=5,
                    label=f'{label} p={p:.2f}')

        ax.set_xlabel('Quality Factor Q')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{label}: PSNR vs Quality Factor')
        ax.legend(loc='lower right', framealpha=0.8, fontsize=8)
        ax.set_ylim(bottom=0)

    fig.suptitle('PSNR vs Quality Factor (by Channel Condition)', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, '07_psnr_vs_quality.png')


def plot_repeat_impact(results: List[Dict], plt_module):
    """
    图 8: 重复编码次数对 PSNR/SSIM 的影响
    """
    import matplotlib.pyplot as plt

    repeat_results = [r for r in results if r['repeat'] in REPEAT_VALUES
                      and r['param'] > 0 and r['quality'] == 50]

    if not repeat_results:
        print("  [SKIP] 无 repeat 对比数据")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, metric in enumerate(['psnr', 'ssim']):
        ax = axes[ax_idx]
        for ch_type, marker, color in [('bsc', 'o', '#3498db'), ('bec', 's', '#e74c3c')]:
            ch_data = [r for r in repeat_results if r['channel'] == ch_type]
            by_rep = aggregate_by(ch_data, 'repeat')
            repeats = sorted(by_rep.keys())
            vals = [np.mean([r[metric] for r in by_rep[r]
                             if r[metric] is not None and r[metric] != float('inf')])
                    for r in repeats]
            ax.plot(repeats, vals, marker=marker, color=color, linewidth=2,
                    markersize=8, label=ch_type.upper())

        ax.set_xlabel('Repeat Count N')
        ax.set_ylabel(metric.upper() + (' (dB)' if metric == 'psnr' else ''))
        ax.set_title(f'{metric.upper()} vs Repeat Encoding')
        ax.legend(framealpha=0.8)
        ax.set_xticks(REPEAT_VALUES)

    fig.suptitle('Impact of Repeat Encoding on Reconstruction Quality', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, '08_repeat_impact.png')


def plot_dashboard(results: List[Dict], plt_module):
    """
    图 9: 综合仪表盘
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    main_results = [r for r in results if r.get('_keep', True)]

    # --- 左上: PSNR heatmap (BSC) ---
    ax1 = fig.add_subplot(2, 3, 1)
    _plot_heatmap(ax1, main_results, 'bsc', 'psnr', 'PSNR (dB) - BSC')

    # --- 右上: PSNR heatmap (BEC) ---
    ax2 = fig.add_subplot(2, 3, 2)
    _plot_heatmap(ax2, main_results, 'bec', 'psnr', 'PSNR (dB) - BEC')

    # --- 中左: SSIM heatmap (BSC) ---
    ax3 = fig.add_subplot(2, 3, 3)
    _plot_heatmap(ax3, main_results, 'bsc', 'ssim', 'SSIM - BSC')

    # --- 中中: Compression Ratio vs Q ---
    ax4 = fig.add_subplot(2, 3, 4)
    by_q = aggregate_by(main_results, 'quality')
    qs = sorted(by_q.keys())
    crs = [mean_std([r['compression_ratio'] for r in by_q[q]
                     if r['compression_ratio'] and r['compression_ratio'] != float('inf')])[0]
           for q in qs]
    ax4.fill_between(qs, 0, crs, alpha=0.3, color='steelblue')
    ax4.plot(qs, crs, 'o-', color='steelblue', linewidth=2.5, markersize=8)
    ax4.set_xlabel('Quality Factor Q')
    ax4.set_ylabel('Compression Ratio')
    ax4.set_title('Compression Ratio')

    # --- 中右: Time breakdown (avg over all runs) ---
    ax5 = fig.add_subplot(2, 3, 5)
    stages = ['Source\nEnc', 'Channel\nEnc', 'Inter\nleave', 'Trans\nmission', 'Channel\nDec', 'Source\nDec']
    keys = ['time_source_enc', 'time_channel_enc', 'time_interleave',
            'time_transmission', 'time_channel_dec', 'time_source_dec']
    colors = plt.cm.Set2(np.linspace(0, 1, len(stages)))
    means = [np.mean([r[k] for r in main_results if r.get(k) is not None and r[k] > 0]) for k in keys]
    means = [max(m, 0) for m in means]
    bars = ax5.barh(stages, means, color=colors, edgecolor='white')
    ax5.set_xlabel('Time (seconds)')
    ax5.set_title('Avg Time per Stage')
    for bar, val in zip(bars, means):
        if val > 0:
            ax5.text(val + max(means) * 0.02, bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}s', va='center', fontsize=8)

    # --- 右下: Summary table ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    _draw_summary_table(ax6, main_results)

    fig.suptitle('ISC Project 2 — Comprehensive Performance Dashboard',
                 fontweight='bold', fontsize=16, y=1.01)
    fig.tight_layout()
    save_figure(fig, '09_dashboard.png')


def _plot_heatmap(ax, results, ch_type, metric, title):
    """在 ax 上绘制 heatmap。"""
    import matplotlib.pyplot as plt

    ch_results = [r for r in results if r['channel'] == ch_type]
    qs = sorted(set(r['quality'] for r in ch_results))
    params = sorted(set(r['param'] for r in ch_results))

    data = np.zeros((len(qs), len(params)))
    for qi, q in enumerate(qs):
        for pi, p in enumerate(params):
            vals = [r[metric] for r in ch_results
                    if r['quality'] == q and r['param'] == p
                    and r[metric] is not None and r[metric] != float('inf')]
            data[qi, pi] = np.mean(vals) if vals else 0

    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', origin='lower')
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([f'{p:.2f}' for p in params], rotation=45, fontsize=8)
    ax.set_yticks(range(len(qs)))
    ax.set_yticklabels(qs, fontsize=8)
    ax.set_xlabel('Error Probability', fontsize=9)
    ax.set_ylabel('Quality Q', fontsize=9)
    ax.set_title(title, fontsize=10)

    # Annotate cells
    for qi in range(len(qs)):
        for pi in range(len(params)):
            val = data[qi, pi]
            text_color = 'white' if val < np.median(data) else 'black'
            if data[qi, pi] > 0:
                ax.text(pi, qi, f'{val:.1f}', ha='center', va='center',
                        fontsize=7, color=text_color, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8)


def _draw_summary_table(ax, results):
    """绘制汇总统计表格。"""
    psnrs = [r['psnr'] for r in results if r['psnr'] is not None and r['psnr'] != float('inf')]
    ssims = [r['ssim'] for r in results if r['ssim'] is not None]
    crs = [r['compression_ratio'] for r in results
           if r['compression_ratio'] and r['compression_ratio'] != float('inf')]
    times = [r['time_source_enc'] + r['time_channel_enc'] + r['time_channel_dec'] + r['time_source_dec']
             for r in results if all(r.get(k) is not None for k in
                                     ['time_source_enc', 'time_channel_enc', 'time_channel_dec', 'time_source_dec'])]

    cell_text = [
        ['PSNR', f'{np.mean(psnrs):.1f} dB', f'{np.min(psnrs):.1f}', f'{np.max(psnrs):.1f}'],
        ['SSIM', f'{np.mean(ssims):.4f}', f'{np.min(ssims):.4f}', f'{np.max(ssims):.4f}'],
        ['Compression', f'{np.mean(crs):.1f}x', f'{np.min(crs):.1f}x', f'{np.max(crs):.1f}x'],
        ['Total Time', f'{np.mean(times):.3f}s', f'{np.min(times):.3f}s', f'{np.max(times):.3f}s'],
    ]

    table = ax.table(cellText=cell_text,
                     colLabels=['Metric', 'Mean', 'Min', 'Max'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[0] % 2 == 0:
            cell.set_facecolor('#f0f0f0')

    ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', y=1.05)


def generate_all_charts(results: List[Dict]):
    """生成所有可视化图表。"""
    print("\n" + "=" * 60)
    print("  生成可视化图表")
    print("=" * 60)

    plt = setup_plot_style()

    chart_funcs = [
        ('PSNR vs Error Rate', plot_psnr_vs_error_rate),
        ('SSIM vs Error Rate', plot_ssim_vs_error_rate),
        ('Compression Ratio', plot_compression_ratio),
        ('Complexity Breakdown', plot_complexity_breakdown),
        ('Rate-Distortion', plot_rate_distortion),
        ('Viterbi Performance', plot_viterbi_performance),
        ('PSNR vs Quality', plot_psnr_vs_quality),
        ('Repeat Encoding Impact', plot_repeat_impact),
        ('Dashboard', plot_dashboard),
    ]

    for name, func in chart_funcs:
        print(f"  {name}...")
        try:
            func(results, plt)
        except Exception as e:
            print(f"    [WARN] {name} 生成失败: {e}")

    print(f"\n所有图表已保存至: {OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════
# CSV 输出
# ═══════════════════════════════════════════════════════════════════════

def save_csv(results: List[Dict], path: str):
    """保存实验结果到 CSV。"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fieldnames = [
        'image', 'channel', 'param', 'quality', 'repeat',
        'psnr', 'ssim', 'compression_ratio',
        'time_source_enc', 'time_channel_enc', 'time_interleave',
        'time_transmission', 'time_channel_dec', 'time_source_dec',
        'actual_error_rate', 'viterbi_ber',
        'source_bits', 'encoded_bits',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"\n实验结果已保存: {path} ({len(results)} 行)")


def load_csv(path: str) -> List[Dict[str, Any]]:
    """从 CSV 加载实验结果。"""
    results: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 类型转换
            for key in ['param', 'quality', 'repeat']:
                if key in row and row[key]:
                    row[key] = int(row[key]) if key in ('quality', 'repeat') else float(row[key])
            for key in ['psnr', 'ssim', 'compression_ratio', 'actual_error_rate',
                        'viterbi_ber', 'source_bits', 'encoded_bits']:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        row[key] = None
            for key in ['time_source_enc', 'time_channel_enc', 'time_interleave',
                        'time_transmission', 'time_channel_dec', 'time_source_dec']:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        row[key] = None
            row['_keep'] = True  # CSV 中的数据都是主实验组
            results.append(row)
    print(f"从 CSV 加载了 {len(results)} 条实验记录: {path}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print("=" * 60)
    print("  ISC Project 2 — 综合性能分析")
    print("=" * 60)
    print(f"  模式: {'快速' if args.quick else '完整'}")
    print(f"  图表: {'跳过' if args.no_charts else '生成'}")
    print()

    # 从 CSV 加载或运行实验
    if args.from_csv:
        if not os.path.isfile(args.from_csv):
            print(f"[ERROR] CSV 文件不存在: {args.from_csv}")
            sys.exit(1)
        results = load_csv(args.from_csv)
    else:
        results = run_analysis(args)
        save_csv(results, args.csv)

    # 生成图表
    if not args.no_charts:
        generate_all_charts(results)

    # 终端汇总
    main_results = [r for r in results if r.get('_keep', True)]
    psnrs = [r['psnr'] for r in main_results if r['psnr'] is not None and r['psnr'] != float('inf')]
    ssims = [r['ssim'] for r in main_results if r['ssim'] is not None]
    crs = [r['compression_ratio'] for r in main_results
           if r['compression_ratio'] and r['compression_ratio'] != float('inf')]

    print()
    print("=" * 60)
    print("  分析汇总")
    print("=" * 60)
    print(f"  实验总数:   {len(main_results)}")
    if psnrs:
        print(f"  PSNR:       {np.mean(psnrs):.1f} dB (mean)  "
              f"[{np.min(psnrs):.1f}, {np.max(psnrs):.1f}]")
    if ssims:
        print(f"  SSIM:       {np.mean(ssims):.4f} (mean)  "
              f"[{np.min(ssims):.4f}, {np.max(ssims):.4f}]")
    if crs:
        print(f"  压缩率:     {np.mean(crs):.1f}x (mean)  "
              f"[{np.min(crs):.1f}, {np.max(crs):.1f}]")
    print(f"  CSV:        {args.csv}")
    if not args.no_charts:
        print(f"  图表:       {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
