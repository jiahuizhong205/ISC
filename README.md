# ISC Project 2 — 图像有损源编码与信道编码

## 项目概述

对给定图集进行**有损源编码**，通过 **BSC / BEC 信道**模拟无线传输，对接收图像质量从准确率、算法复杂度、PSNR 三个维度进行评估。

- **编程语言**：Python 3.10+

---

## 项目目录结构

```
ISC/
├── README.md
├── requirements.txt              # Python依赖
├── data/
│   └── kodak/                    # 柯达标准测试图集 (24张)
├── src/
│   ├── source_coding/
│   │   ├── encoder.py            # DCT变换编码器
│   │   └── decoder.py            # DCT变换解码器
│   ├── channel_coding/
│   │   ├── convolutional.py      # 卷积码编码 + Viterbi译码
│   │   └── interleaver.py        # 交织/解交织(BEC抗突发删除)
│   ├── channel_model/
│   │   └── channel.py            # BSC & BEC 信道仿真
│   └── main.py                   # 系统集成流水线
├── scripts/
│   └── run_experiments.py        # 批量实验脚本
├── results/
│   └── figures/                  # PSNR/复杂度曲线输出
├── report/
│   └── report.pdf                # 英文报告
└── output/                       # 中间输出/日志
```

---

## 技术选型

### 1. 图集 — 柯达无损真彩图集 (Kodak Lossless True Color Image Suite)

- 24 张标准 768×512（或 512×768）真彩图像
- 图像压缩领域使用最广泛的公开标准测试集之一，便于 PSNR 横向对比
- 下载地址：`https://r0k.us/graphics/kodak/`

### 2. 源编码 — 基于 DCT 的变换编码（类 JPEG 基线）

- **分块**：8×8 像素块
- **变换**：二维 DCT
- **量化**：亮度/色度量化表（可调 quality factor）
- **熵编码**：Zigzag 扫描 + 游程编码 + Huffman 编码

| 选择原因 | 说明 |
|---|---|
| 理论成熟 | 课程涉及的变换编码范式最直观的实现 |
| 复杂度可控 | quality factor 直接调节压缩率 ↔ PSNR 权衡 |
| 有损可控 | 量化步骤天然支持有损编码需求 |

### 3. 信道编码 — 卷积码 + Viterbi 译码

- **码型**：(2, 1, K) 卷积码，约束长度 K=7
- **译码**：硬判决/软判决 Viterbi 算法
- **可选辅码**：CRC-16 校验（辅助检测残留错误）

| 信道 | 处理方式 |
|---|---|
| **BSC** | 卷积码直接纠错，Viterbi 硬/软判决译码 |
| **BEC** | 交织器打散删除簇 + Viterbi 擦除译码 |

> 备选方案：若卷积码效果不理想，可降级为 (7,4) 汉明码 + 块交织，实现简单但纠错能力有限。

### 4. 信道模型

| 参数 | BSC | BEC |
|---|---|---|
| 模型 | 二进制对称信道 | 二进制删除信道 |
| 可变参数 | 交叉概率 ε | 删除概率 p |
| 仿真方式 | 以概率 ε 翻转比特 | 以概率 p 擦除比特(标记为 `None`) |
| 参数范围 | ε ∈ {0, 0.01, 0.05, 0.1} | p ∈ {0, 0.05, 0.1, 0.2} |

### 5. 评估指标

| 指标 | 计算方法 | 说明 |
|---|---|---|
| **PSNR** | 10·log₁₀(255² / MSE) | 像素级重建质量，越高越好 |
| **SSIM** | 结构相似度 | 感知质量，补充 PSNR 的不足 |
| **算法复杂度** | 编码/解码耗时、大 O 分析 | 分别统计源编解码和信道编解码的时间 |
| **压缩率** | 原始大小 / 压缩后大小 | 压缩效率 |

---

## 四人分工

| 角色 | 负责模块 | 核心任务 |
|---|---|---|
| **成员 A** | 源编码 | DCT 变换、量化、熵编码、源解码；输出压缩比特流 |
| **成员 B** | 信道编码 | 卷积码编解码器、Viterbi 译码、交织器；适配 BSC/BEC |
| **成员 C** | 信道模型 + 系统集成 | BSC/BEC 仿真、拼接 A→B→C 流水线、批量实验 |
| **成员 D** | 性能分析 + 英文报告 | PSNR/SSIM/复杂度计算、可视化图表、英文报告撰写、最终打包提交 |

### 模块接口约定

```
源编码输出 (A → B)
  ── 比特流 (list[int] / bytes)

信道编码输出 (B → C)
  ── 编码比特流 (list[int])

信道传输 (C)
  BSC: 翻转比特  → 受损比特流 (list[int])
  BEC: 删除比特  → 受损比特流 + 删除位置标记 (list[int | None])

信道译码输出 (C → A's decoder)
  ── 修正后的比特流 (list[int] / bytes)
```

> **约定**：所有模块间数据交换统一使用 Python `list[int]` 或 `numpy.ndarray`，方便调试和可视化。

---

## 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

`requirements.txt` 内容：
```
numpy>=1.24
scipy>=1.10
opencv-python>=4.8
matplotlib>=3.7
pillow>=10.0
```

---

## 使用方式

```bash
# 单次测试运行
python src/main.py --image data/kodak/kodim01.png --ber 0.05 --channel bsc

# 批量实验
python scripts/run_experiments.py

# 生成性能图表
python scripts/run_experiments.py --plot
```

---

## 时间规划建议

| 阶段 | 内容 | 建议耗时 |
|---|---|---|
| 第 1 周 | 确定技术方案，下载图集，搭建项目骨架 | 2 天 |
| 第 2 周 | A/B/C 各自实现模块，单元测试 | 4 天 |
| 第 3 周 | C 完成系统集成，D 跑实验收集数据 | 3 天 |
| 第 4 周 | D 撰写报告，全员审查，打包提交 | 3 天 |
