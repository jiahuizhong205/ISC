# ISC Project 2 — 图像有损源编码与信道编码

**课程:** NJU 2026 春 — 信息论基础
**团队成员:** 张海洋 (源编码), 冉丽滢 (信道编码), 仲嘉辉 (系统集成), 陈玉熙 (评估与报告)
**日期:** 2026 年 6 月

---

## 项目概述

设计并实现完整的端到端图像编码传输系统。对柯达图集进行 DCT 有损源编码，通过 (2,1,7) 卷积码和块交织器保护，经 BSC/BEC 信道模拟无线传输，Viterbi 译码纠错后重建图像。从**准确率 (Accuracy)、算法复杂度 (Algorithm Complexity)、峰值信噪比 (PSNR)** 三个维度评估系统性能。

- **编程语言：** Python 3.10+

---

## 项目目录结构

```
ISC/
├── README.md                        # 项目说明
├── CONFIG.md                        # 配置文件说明
├── IMPROVEMENTS.md                  # 容错改进记录
├── config.json                      # 运行参数配置
├── requirements.txt                 # Python 依赖
├── data/
│   └── kodim01~12.png               # 柯达标准测试图集 (12 张)
├── src/
│   ├── interfaces.py                # 模块接口抽象基类
│   ├── main.py                      # 端到端流水线
│   ├── bitstream.py                 # 比特流打包/拆包
│   ├── source_coding/
│   │   ├── encoder.py               # DCT 编码器 (含 Huffman + 重复编码)
│   │   ├── decoder.py               # DCT 解码器 (含多数投票容错)
│   │   └── test_source_coding.py    # 源编码单元测试
│   ├── channel_coding/
│   │   ├── convolutional.py         # (2,1,7) 卷积码 + Viterbi 译码
│   │   ├── interleaver.py           # 块交织/解交织器
│   │   └── README.md
│   └── channel_model/
│       └── channel.py               # BSC / BEC 信道仿真
├── scripts/
│   ├── run_experiments.py           # 批量实验 + CSV 导出
│   ├── run_full_experiments.py      # 完整实验套件
│   ├── analysis.py                  # 综合性能分析
│   ├── plot_advanced.py             # 高级图表绘制
│   └── md2docx.py                   # Markdown → Word 转换
├── results/
│   ├── analysis.csv                 # 实验数据 (288 组)
│   ├── analysis_kodak.csv           # 柯达图集数据
│   ├── ANALYSIS_REPORT.md           # 数据分析报告
│   └── figures/                     # 16 张性能图表
├── report/
│   ├── report_en.md                 # 英文报告
│   ├── report_en.docx               # 英文报告 (Word)
│   ├── report_cn.md                 # 中文报告
│   └── report_cn.docx               # 中文报告 (Word)
├── tests/
│   └── test_channel_coding.py       # 信道编码单元测试
└── output/                          # 运行输出
```

---

## 技术规格

| 组件 | 算法 | 关键参数 |
|---|---|---|
| 色彩变换 | ITU-R BT.601 YCbCr | 4:4:4 采样 |
| 分块 | 8×8 像素 | — |
| 变换 | 二维 DCT-II (SciPy) | 正交归一 |
| 量化 | JPEG 标准量化表 | Quality factor Q ∈ [1, 100] |
| 熵编码 | Zigzag → RLE → 全局 Huffman | 块级独立编址 |
| 容错机制 | N× 重复 + 多数投票 | N ∈ {1, 3, 5}，可配置 |
| 信道编码 | (2,1,7) 卷积码 | 生成多项式 (171, 133) 八进制 |
| 译码 | Viterbi 硬判决 (BSC) / 软判决 (BEC) | K=7, 64 状态 |
| 交织 | 块交织 64×128 | 按行写入、按列读出 |
| 信道 | BSC / BEC | ε ∈ [0, 0.1], p ∈ [0, 0.2] |

---

## 实验设计

| 参数 | 取值 |
|---|---|
| 测试图像 | 12 张柯达无损真彩图集 (768×512 或 512×768, RGB) |
| 质量因子 Q | 10, 50, 90 |
| 重复编码 N | 1 (基线) |
| BSC 误码率 ε | 0, 0.01, 0.05, 0.10 |
| BEC 删除率 p | 0, 0.05, 0.10, 0.20 |
| 随机种子 | 42 |
| 总实验数 | 288 |

---

## 评估指标

### 1. 准确率 (Accuracy)

| 指标 | 计算方式 | 说明 |
|---|---|---|
| 信道 BER | 传输中翻转/删除的 bit 比例 | 信道原始错误水平 |
| 残留 BER | Viterbi 译码后错误 bit 比例 | 纠错后剩余错误 |
| 压缩率 | 原始图像 bit ÷ 压缩后 bit | >1 表示有压缩 |

### 2. 算法复杂度 (Algorithm Complexity)

| 环节 | 算法 | 大 O 复杂度 |
|---|---|---|
| 源编码 | DCT + 量化 + RLE + Huffman | O(N), N=像素数 |
| 信道编码 | (2,1,7) 卷积码 | O(N), N=源 bit 数 |
| 交织 | 块交织 (64×128) | O(N) |
| 信道译码 | Viterbi 算法 | O(N·2ᴷ), K=7 |
| 源解码 | Huffman + IDCT | O(N) |

> Viterbi 译码占总耗时约 80%，是端到端流水线的性能瓶颈。

### 3. 峰值信噪比 (PSNR)

| 公式 | 说明 |
|---|---|
| PSNR = 10·log₁₀(255² / MSE) | 像素级保真度 |

| PSNR 范围 | 质量等级 |
|---|---|
| ∞ | 无损重建 |
| ≥ 30 dB | 优秀 — 肉眼不可察觉差异 |
| 25 ~ 30 dB | 良好 — 轻微失真 |
| 20 ~ 25 dB | 可接受 — 有可见噪声 |
| < 20 dB | 较差或严重损坏 |

### 补充指标: SSIM (结构相似度)

| 公式 | 值域 | 说明 |
|---|---|---|
| Wang et al. (2004) | [0, 1] | 1 表示完全相同；综合考虑亮度、对比度、结构 |

> PSNR 与 SSIM 在所有实验条件下呈高度正相关 (R² > 0.9)。

---

## 核心实验结果

### 无损条件下 (ε=0, p=0)

| Q 值 | PSNR 范围 | 平均 PSNR | 平均压缩率 |
|---|---|---|---|
| Q=10 | 23.8 ~ 28.9 dB | 27.1 dB | 29.6× |
| Q=50 | 29.8 ~ 35.4 dB | 33.0 dB | 12.2× |
| Q=90 | 37.4 ~ 41.6 dB | 39.6 dB | 5.2× |

### 有损条件下 (N=1, Q=50)

| 信道条件 | 平均 PSNR | 说明 |
|---|---|---|
| BSC ε=0.01 | 10.6 dB | 严重损坏 |
| BSC ε=0.05 | 8.1 dB | 灾难性 |
| BEC p=0.05 | 17.9 dB | 中等 |
| BEC p=0.10 | 12.8 dB | 严重 |

### 关键发现

1. **硬判决 Viterbi 在 BSC 上效果不佳** — 在 ε ≥ 1% 时，译码后 BER 反而略高于信道原始 BER。这是硬判决译码在低 SNR 下的固有局限。
2. **软判决 Viterbi 在 BEC 上非常有效** — 利用已知擦除位置信息，BER 改善 9~38 倍。
3. **BEC 始终优于 BSC** — 在相同错误概率下，BEC 的 PSNR 约高出 **10 dB**。
4. **Viterbi 译码是性能瓶颈** — 占端到端总耗时约 80%。
5. **重复编码 (N=5) 可补偿 BEC 信道错误** — 以 5× 带宽换取接近无损的重建质量。

---

## 四人分工

| 成员 | 角色 | 核心任务 |
|---|---|---|
| 张海洋 | 源编码 (A) | DCT 变换、量化、Huffman 熵编解码、N× 重复容错编码 |
| 冉丽滢 | 信道编码 (B) | (2,1,7) 卷积码、Viterbi 译码、块交织器 |
| 仲嘉辉 | 系统集成 (C) | BSC/BEC 信道仿真、流水线拼接、配置文件、批量实验 |
| 陈玉熙 | 评估与报告 (D) | PSNR/SSIM/复杂度分析、16 张图表、中英文报告 |

---

## 环境配置

```bash
pip install -r requirements.txt
```

依赖项: `numpy>=1.24`, `scipy>=1.10`, `opencv-python>=4.8`, `matplotlib>=3.7`, `pillow>=10.0`

---

## 使用方式

```bash
# 单次运行 (使用配置文件 config.json)
python -m src.main

# 单次运行 (命令行覆盖参数)
python -m src.main --param 0.01 --repeat 5

# 批量实验
python scripts/run_experiments.py --channels bsc --subset 1

# 完整实验套件 (~20 min, 288 组)
python scripts/run_full_experiments.py --medium --no-repeats

# 快速验证 (~5 min, 96 组)
python scripts/run_full_experiments.py --quick --no-repeats

# 生成分析图表 (从已有 CSV)
python scripts/plot_advanced.py --csv results/analysis.csv

# 综合分析与绘图
python scripts/analysis.py --from-csv results/analysis.csv

# 单元测试
python src/source_coding/test_source_coding.py
python tests/test_channel_coding.py
```

---

## 参考文献

1. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Trans. Image Processing*, 13(4), 600–612.
2. Viterbi, A. J. (1967). Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. *IEEE Trans. Information Theory*, 13(2), 260–269.
3. ITU-R BT.601-7 (2011). Studio encoding parameters of digital television.
4. Pennebaker, W. B., & Mitchell, J. L. (1992). *JPEG: Still Image Data Compression Standard*. Springer.
