#!/usr/bin/env python
"""
ISC Project 2 — 补充可视化分析

从 experiments CSV 读取数据，生成补充的专业图表:
  1. PSNR vs SSIM 相关性散点图
  2. 压缩率 vs 信道错误率的关系
  3. Viterbi 纠错改善因子热力图
  4. 各阶段时间复杂度验证 (time vs bits)
  5. 分图像类型质量对比
  6. Quality-Repeat 交互效应

用法:
    python scripts/plot_advanced.py                        # 从默认 CSV
    python scripts/plot_advanced.py --csv results/analysis.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

# matplotlib 配置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTPUT_DIR = 'results/figures'


def load_csv(path: str) -> List[Dict[str, Any]]:
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            for k in ['param', 'quality', 'repeat']:
                if k in row and row[k]:
                    row[k] = int(row[k]) if k in ('quality', 'repeat') else float(row[k])
            for k in ['psnr', 'ssim', 'compression_ratio', 'actual_error_rate',
                       'viterbi_ber', 'source_bits', 'encoded_bits']:
                if k in row and row[k]:
                    try: row[k] = float(row[k])
                    except (ValueError, TypeError): row[k] = None
            for k in ['time_source_enc', 'time_channel_enc', 'time_interleave',
                       'time_transmission', 'time_channel_dec', 'time_source_dec']:
                if k in row and row[k]:
                    try: row[k] = float(row[k])
                    except (ValueError, TypeError): row[k] = None
            results.append(row)
    return results


def save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  [OK] {path}")


# ═══════════════════════════════════════════════════════════════════════
# 图 10: PSNR vs SSIM 相关性分析
# ═══════════════════════════════════════════════════════════════════════

def plot_psnr_ssim_correlation(results: List[Dict]):
    """PSNR 与 SSIM 的相关性散点图，展示两种质量指标的对应关系。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'bsc': '#3498db', 'bec': '#e74c3c'}
    markers = {10: 'o', 50: 's', 90: 'D'}

    for ax_idx, ch_type in enumerate(['bsc', 'bec']):
        ax = axes[ax_idx]
        ch_data = [r for r in results if r['channel'] == ch_type]

        for q, marker in markers.items():
            q_data = [r for r in ch_data if r['quality'] == q
                      and r['psnr'] is not None and r['psnr'] != float('inf')
                      and r['ssim'] is not None]
            if not q_data:
                continue
            psnrs = [r['psnr'] for r in q_data]
            ssims = [r['ssim'] for r in q_data]

            ax.scatter(psnrs, ssims, c=colors[ch_type], marker=marker,
                       s=40, alpha=0.6, edgecolors='white', linewidth=0.3,
                       label=f'Q={q}')

        # 拟合线
        all_psnr = [r['psnr'] for r in ch_data
                    if r['psnr'] is not None and r['psnr'] != float('inf')
                    and r['ssim'] is not None]
        all_ssim = [r['ssim'] for r in ch_data
                    if r['psnr'] is not None and r['psnr'] != float('inf')
                    and r['ssim'] is not None]
        if len(all_psnr) > 2:
            z = np.polyfit(all_psnr, all_ssim, 2)
            x_smooth = np.linspace(min(all_psnr), max(all_psnr), 100)
            y_smooth = np.polyval(z, x_smooth)
            ax.plot(x_smooth, y_smooth, '--', color='#555555', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('PSNR (dB)')
        ax.set_ylabel('SSIM')
        ax.set_title(f'{ch_type.upper()}: PSNR vs SSIM Correlation')
        ax.legend(framealpha=0.8)

    fig.suptitle('PSNR vs SSIM Correlation Analysis', fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, '10_psnr_ssim_correlation.png')


# ═══════════════════════════════════════════════════════════════════════
# 图 11: 压缩率 vs 信道错误率的关系
# ═══════════════════════════════════════════════════════════════════════

def plot_compression_vs_error(results: List[Dict]):
    """展示不同压缩率下对信道错误的鲁棒性。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, ch_type in enumerate(['bsc', 'bec']):
        ax = axes[ax_idx]
        ch_data = [r for r in results if r['channel'] == ch_type
                   and r['param'] > 0]

        by_param = defaultdict(list)
        for r in ch_data:
            if r['compression_ratio'] and r['compression_ratio'] != float('inf'):
                by_param[r['param']].append(r)

        params = sorted(by_param.keys())
        cmap = plt.cm.RdYlGn_r
        norm = plt.Normalize(min(params), max(params))

        for p in params:
            crs = [r['compression_ratio'] for r in by_param[p]]
            psnrs = [r['psnr'] for r in by_param[p]
                     if r['psnr'] is not None and r['psnr'] != float('inf')]
            if crs and psnrs:
                ax.scatter(crs, psnrs, c=[cmap(norm(p))], s=30,
                           alpha=0.5, edgecolors='none')

        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{ch_type.upper()}: PSNR vs Compression Ratio')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label(f'{ch_type.upper()} Error Prob.', fontsize=9)

    fig.suptitle('Compression vs Error Resilience Trade-off', fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, '11_compression_vs_error.png')


# ═══════════════════════════════════════════════════════════════════════
# 图 12: Viterbi 纠错改善因子热力图
# ═══════════════════════════════════════════════════════════════════════

def plot_viterbi_improvement(results: List[Dict]):
    """Viterbi 译码纠错改善倍数 = 信道BER / 残留BER。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, ch_type in enumerate(['bsc', 'bec']):
        ax = axes[ax_idx]
        ch_data = [r for r in results if r['channel'] == ch_type
                   and r['actual_error_rate'] is not None
                   and r['actual_error_rate'] > 0
                   and r['viterbi_ber'] is not None]

        by_pq = defaultdict(list)
        for r in ch_data:
            by_pq[(r['param'], r['quality'])].append(r)

        unique_params = sorted(set(r['param'] for r in ch_data))
        unique_qs = sorted(set(r['quality'] for r in ch_data))

        data = np.zeros((len(unique_qs), len(unique_params)))
        for qi, q in enumerate(unique_qs):
            for pi, p in enumerate(unique_params):
                vals = [r['actual_error_rate'] / max(r['viterbi_ber'], 1e-10)
                        for r in ch_data
                        if r['param'] == p and r['quality'] == q
                        and r['viterbi_ber'] and r['viterbi_ber'] > 0]
                data[qi, pi] = np.mean(vals) if vals else 0

        im = ax.imshow(data, aspect='auto', cmap='YlOrRd', origin='lower')
        ax.set_xticks(range(len(unique_params)))
        ax.set_xticklabels([f'{p:.3f}' for p in unique_params], rotation=45, fontsize=8)
        ax.set_yticks(range(len(unique_qs)))
        ax.set_yticklabels(unique_qs, fontsize=8)
        ax.set_xlabel(f'{ch_type.upper()} Error Probability', fontsize=9)
        ax.set_ylabel('Quality Q', fontsize=9)
        ax.set_title(f'{ch_type.upper()}: Viterbi Improvement Factor')

        for qi in range(len(unique_qs)):
            for pi in range(len(unique_params)):
                val = data[qi, pi]
                if val > 0:
                    ax.text(pi, qi, f'{val:.0f}x', ha='center', va='center',
                            fontsize=7, fontweight='bold',
                            color='white' if val < np.median(data) else 'black')

        plt.colorbar(im, ax=ax, shrink=0.8, label='Improvement Factor')

    fig.suptitle('Viterbi Decoder: Error Correction Improvement Factor',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, '12_viterbi_improvement_heatmap.png')


# ═══════════════════════════════════════════════════════════════════════
# 图 13: 时间复杂度分析 (time vs input size)
# ═══════════════════════════════════════════════════════════════════════

def plot_complexity_scaling(results: List[Dict]):
    """验证各阶段时间复杂度 O(N) 的理论预测。"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    stages = [
        ('time_source_enc', 'Source Encoding (DCT+RLE+Huffman)', axes[0, 0]),
        ('time_channel_enc', 'Channel Encoding (Conv Code)', axes[0, 1]),
        ('time_interleave', 'Interleaving (Block)', axes[0, 2]),
        ('time_channel_dec', 'Channel Decoding (Viterbi)', axes[1, 0]),
        ('time_source_dec', 'Source Decoding (Huffman+IDCT)', axes[1, 1]),
    ]

    clean_data = [r for r in results
                  if r['param'] == 0.0  # 无噪声情况，避免错误传播影响
                  and r.get('source_bits') and r.get('encoded_bits')]

    for stage_key, title, ax in stages:
        xs = []
        ys = []
        for r in clean_data:
            bit_count = r.get('source_bits', 0)
            elapsed = r.get(stage_key)
            if bit_count and elapsed is not None and elapsed > 0:
                xs.append(bit_count)
                ys.append(elapsed)

        if xs:
            ax.scatter(xs, ys, c='#3498db', s=25, alpha=0.5, edgecolors='none')

            # 线性拟合
            z = np.polyfit(xs, ys, 1)
            x_fit = np.linspace(min(xs), max(xs), 100)
            y_fit = np.polyval(z, x_fit)
            ax.plot(x_fit, y_fit, '-', color='#e74c3c', linewidth=2,
                    label=f'O(N): {z[0]:.2e} s/bit')

            ax.set_xlabel('Source Bits')
            ax.set_ylabel('Time (s)')
            ax.set_title(title)
            ax.legend(fontsize=7)
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    # 第6格: 总耗时对比
    ax = axes[1, 2]
    stage_names = ['Src Enc', 'Ch Enc', 'Interleave', 'Ch Dec', 'Src Dec']
    stage_keys = ['time_source_enc', 'time_channel_enc', 'time_interleave',
                  'time_channel_dec', 'time_source_dec']
    total_times = []
    for k in stage_keys:
        times = [r[k] for r in clean_data if r.get(k) is not None and r[k] > 0]
        total_times.append(np.mean(times) if times else 0)

    colors = plt.cm.Set2(np.linspace(0, 1, len(stage_names)))
    bars = ax.bar(range(len(stage_names)), total_times, color=colors, edgecolor='white')
    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, fontsize=8)
    ax.set_ylabel('Avg Time (s)')
    ax.set_title('Avg Time per Stage (error-free)')
    for bar, val in zip(bars, total_times):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(total_times) * 0.02,
                    f'{val:.3f}s', ha='center', fontsize=7)

    fig.suptitle('Algorithm Complexity: Time vs Input Size', fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, '13_complexity_scaling.png')


# ═══════════════════════════════════════════════════════════════════════
# 图 14: 分图像类型的质量对比
# ═══════════════════════════════════════════════════════════════════════

def plot_per_image_quality(results: List[Dict]):
    """不同图像类型（自然/几何/渐变/噪声）的压缩质量对比。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    images = sorted(set(r['image'] for r in results))
    qualities = sorted(set(r['quality'] for r in results))
    ch_types = ['bsc', 'bec']

    x = np.arange(len(images))
    width = 0.2

    for ax_idx, (ch_type, metric) in enumerate([
        ('bsc', 'psnr'), ('bec', 'psnr'), ('bsc', 'ssim'), ('bec', 'ssim')]):
        ax = axes[ax_idx // 2][ax_idx % 2]
        ch_data = [r for r in results if r['channel'] == ch_type and r['param'] == 0.0]

        for qi, q in enumerate(qualities[:4]):  # 最多4个quality
            q_data = [r for r in ch_data if r['quality'] == q]
            by_img = defaultdict(list)
            for r in q_data:
                short_name = r['image'].replace('synthetic_', '')[:8]
                by_img[short_name].append(r[metric] if r[metric] != float('inf') else 60)

            img_order = sorted(by_img.keys())
            means = [np.mean([v for v in by_img[img] if v is not None]) for img in img_order]

            offset = (qi - len(qualities) / 2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, label=f'Q={q}',
                          alpha=0.8, edgecolor='white', linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(img_order, fontsize=8, rotation=20)
        ax.set_ylabel(metric.upper() + (' (dB)' if metric == 'psnr' else ''))
        ax.set_title(f'{ch_type.upper()}: {metric.upper()} by Image Type (p=0)')
        ax.legend(fontsize=7)

    fig.suptitle('Image Type Quality Comparison (Error-Free)', fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, '14_per_image_quality.png')


# ═══════════════════════════════════════════════════════════════════════
# 图 15: 质量因子与重复编码的交互效应
# ═══════════════════════════════════════════════════════════════════════

def plot_quality_repeat_interaction(results: List[Dict]):
    """展示 Q 和 Repeat 对最终质量的联合影响。"""
    repeat_data = [r for r in results if r['param'] > 0
                   and r['psnr'] is not None and r['psnr'] != float('inf')]

    if not repeat_data:
        print("  [SKIP] 无 repeat 交互数据")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    repeats = sorted(set(r['repeat'] for r in repeat_data))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(repeats)))
    markers = ['o', 's', 'D']

    for ax_idx, (ch_type, metric) in enumerate([('bsc', 'psnr'), ('bsc', 'ssim')]):
        ax = axes[ax_idx]
        ch_data = [r for r in repeat_data if r['channel'] == ch_type]

        for ri, rep in enumerate(repeats):
            rep_data = [r for r in ch_data if r['repeat'] == rep]
            by_q = defaultdict(list)
            for r in rep_data:
                by_q[r['quality']].append(r[metric])

            qs = sorted(by_q.keys())
            means = [np.mean(by_q[q]) for q in qs]

            ax.plot(qs, means, marker=markers[ri % len(markers)],
                    color=colors[ri], linewidth=2, markersize=7,
                    label=f'Repeat={rep}')

        ax.set_xlabel('Quality Factor Q')
        ax.set_ylabel(metric.upper() + (' (dB)' if metric == 'psnr' else ''))
        ax.set_title(f'BSC (p=0.05): {metric.upper()} vs Q by Repeat Count')
        ax.legend(framealpha=0.8)

    fig.suptitle('Interaction: Quality Factor × Repeat Encoding', fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, '15_quality_repeat_interaction.png')


# ═══════════════════════════════════════════════════════════════════════
# 图 16: 信道残差分析 (发送 vs 接收 bit 状态)
# ═══════════════════════════════════════════════════════════════════════

def plot_channel_residual_analysis(results: List[Dict]):
    """分析 Viterbi 译码后残留 BER 与信道参数的关系。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, (ch_type, label) in enumerate([('bsc', 'BSC'), ('bec', 'BEC')]):
        ax = axes[ax_idx]
        ch_data = [r for r in results if r['channel'] == ch_type
                   and r['param'] > 0
                   and r['actual_error_rate'] is not None
                   and r['viterbi_ber'] is not None]

        by_q = defaultdict(list)
        for r in ch_data:
            by_q[r['quality']].append(r)

        colors = plt.cm.tab10(np.linspace(0, 1, len(by_q)))

        for qi, (q, q_data) in enumerate(sorted(by_q.items())):
            by_param = defaultdict(list)
            for r in q_data:
                by_param[r['param']].append(r)

            params = sorted(by_param.keys())
            ch_bers = [np.mean([r['actual_error_rate'] for r in by_param[p]]) for p in params]
            res_bers = [np.mean([r['viterbi_ber'] for r in by_param[p]]) for p in params]

            # 绘制信道BER -> 残留BER 的映射线
            for p, ch_ber, res_ber in zip(params, ch_bers, res_bers):
                ax.plot([ch_ber, ch_ber], [res_ber, ch_ber],
                        color=colors[qi], linewidth=0.8, alpha=0.4)

            ax.scatter(ch_bers, res_bers, color=colors[qi], s=40,
                       marker='o', edgecolors='white', linewidth=0.3,
                       label=f'Q={q}', zorder=5)

        # 对角线 (无纠错改善)
        lims = [1e-4, 1]
        ax.plot(lims, lims, '--', color='gray', linewidth=1, alpha=0.5,
                label='No improvement')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Channel BER (before Viterbi)')
        ax.set_ylabel('Residual BER (after Viterbi)')
        ax.set_title(f'{label}: Before vs After Viterbi')
        ax.legend(fontsize=7, framealpha=0.8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    fig.suptitle('Viterbi Decoder: Channel BER → Residual BER', fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, '16_channel_residual_analysis.png')


# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ISC Project 2 — 补充可视化')
    parser.add_argument('--csv', type=str, default='results/analysis.csv')
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[ERROR] CSV 文件不存在: {args.csv}")
        print(f"  请先运行: python scripts/analysis.py --quick")
        sys.exit(1)

    print("=" * 60)
    print("  ISC Project 2 — 补充图表生成")
    print("=" * 60)

    results = load_csv(args.csv)
    print(f"加载 {len(results)} 条记录\n")

    chart_funcs = [
        ('PSNR-SSIM Correlation', plot_psnr_ssim_correlation),
        ('Compression vs Error', plot_compression_vs_error),
        ('Viterbi Improvement Heatmap', plot_viterbi_improvement),
        ('Complexity Scaling', plot_complexity_scaling),
        ('Per-Image Quality', plot_per_image_quality),
        ('Quality-Repeat Interaction', plot_quality_repeat_interaction),
        ('Channel Residual Analysis', plot_channel_residual_analysis),
    ]

    for name, func in chart_funcs:
        print(f"  {name}...")
        try:
            func(results)
        except Exception as e:
            import traceback
            print(f"    [WARN] {name}: {e}")
            traceback.print_exc()

    print(f"\n所有补充图表已保存至: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
