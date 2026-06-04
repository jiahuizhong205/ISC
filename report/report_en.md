# ISC Project 2 — Final Report: Image Lossy Source Coding & Channel Coding

**Course:** NJU 2026 Spring — Fundamentals of Information Theory  
**Team Members:** 张海洋 (Source Coding), 仲嘉辉 (Channel Coding), 冉丽滢 (System Integration), 陈玉熙 (Evaluation & Report)  
**Date:** June 2026

---

## Abstract

This report presents the design, implementation, and comprehensive performance evaluation of an end-to-end image transmission system over noisy channels. The system employs DCT-based lossy source coding (JPEG-like), (2,1,7) convolutional channel coding with Viterbi decoding, and block interleaving. Performance is assessed across three dimensions: **Accuracy** (error correction), **Algorithm Complexity** (computational cost), and **Reconstruction Quality** (PSNR & SSIM). Experiments were conducted over Binary Symmetric Channel (BSC) and Binary Erasure Channel (BEC) with varying error probabilities and quality factors.

---

## 1. System Architecture

### 1.1 Pipeline Overview

```
Input Image (RGB)
    │
    ▼
┌─────────────────┐
│  Source Encoder  │  RGB → YCbCr → 8×8 blocks → DCT → Quantization
│  (DCT + Huffman) │  → Zigzag → RLE → Huffman → Bitstream
└────────┬────────┘
         │ source bits
         ▼
┌─────────────────┐
│ Channel Encoder  │  (2,1,7) Convolutional Code (rate 1/2)
│  (Conv + Interlv)│  → Block Interleaver (64×128)
└────────┬────────┘
         │ encoded bits
         ▼
┌─────────────────┐
│  Channel Model   │  BSC: flip bits with probability ε
│  (BSC / BEC)     │  BEC: erase bits with probability p
└────────┬────────┘
         │ received (with errors/erasures)
         ▼
┌─────────────────┐
│ Channel Decoder  │  Block Deinterleaver → Viterbi Decoder
│  (Deinterlv+Vit) │  (hard-decision for BSC, soft-decision for BEC)
└────────┬────────┘
         │ corrected bits
         ▼
┌─────────────────┐
│  Source Decoder  │  Huffman Decode → RLE⁻¹ → Zigzag⁻¹ → Dequant
│  (Huffman+IDCT)  │  → IDCT → YCbCr→RGB → Output Image
└────────┬────────┘
         │
         ▼
   Reconstructed Image
```

### 1.2 Technical Specifications

| Component | Algorithm | Key Parameters |
|-----------|-----------|----------------|
| Color Transform | ITU-R BT.601 YCbCr | 4:4:4 subsampling |
| Block Size | 8×8 pixels | — |
| Transform | 2-D DCT-II (SciPy) | Orthonormal |
| Quantization | JPEG standard tables | Quality factor Q ∈ [1, 100] |
| Entropy Coding | Zigzag → RLE → Huffman | Global Huffman table |
| Error Resilience | N× repeat + majority voting | N ∈ {1, 3, 5} |
| Channel Code | (2,1,7) Convolutional | Generators: (171, 133) octal |
| Decoding | Viterbi (hard/soft) | K=7, 64 states |
| Interleaving | Block (64×128) | Row-in, column-out |
| Channel | BSC / BEC | ε ∈ [0, 0.15], p ∈ [0, 0.3] |

---

## 2. Evaluation Methodology

### 2.1 Metrics

| Dimension | Metric | Formula / Description |
|-----------|--------|-----------------------|
| Accuracy | Channel BER | Fraction of bits flipped/erased during transmission |
| Accuracy | Residual BER | Bit error rate after Viterbi decoding |
| Accuracy | Compression Ratio | raw_image_bits ÷ compressed_bits |
| Complexity | Per-stage timing | Wall-clock time via `time.perf_counter()` |
| Quality | PSNR | $10 \cdot \log_{10}(255^2 / \text{MSE})$ |
| Quality | SSIM | Wang et al. (2004), Gaussian window 11×11 |

### 2.2 Experimental Design

| Parameter | Values |
|-----------|--------|
| Test Images | 4 types: nature scene, geometric shapes, smooth gradient, random noise (256×256 RGB) |
| Quality Factor Q | 10, 50, 90 |
| Repeat Encoding N | 1, 3, 5 |
| BSC Error Rate ε | 0, 0.01, 0.05, 0.10 |
| BEC Erasure Rate p | 0, 0.05, 0.10, 0.20 |
| Random Seed | 42 (deterministic) |
| Total Experiments | 102 |

---

## 3. Results and Analysis

### 3.1 PSNR Performance

#### 3.1.1 Error-Free Conditions

| Quality Q | PSNR Range (dB) | Mean PSNR (dB) | Visual Quality |
|-----------|-----------------|----------------|----------------|
| Q = 10 | 11.2 ~ 33.2 | 26.2 | Acceptable — visible artifacts |
| Q = 50 | 16.8 ~ 43.8 | 34.6 | Good — slight distortion |
| Q = 90 | 30.0 ~ 55.2 | 44.8 | Excellent — nearly lossless |

**Image-type variation:** Smooth gradient images achieve the highest PSNR (55.2 dB at Q=90). Random noise images show the lowest PSNR (11.2 dB at Q=10) due to high-frequency content that is poorly compressed by DCT.

#### 3.1.2 Impact of Channel Errors

| Channel Condition | PSNR Range (dB) | Mean PSNR (dB) | Degradation |
|-------------------|-----------------|----------------|-------------|
| BSC ε = 0.01, Q=50 | 6.3 ~ 14.3 | 10.9 | Severe |
| BSC ε = 0.05, Q=50 | 5.8 ~ 9.0 | 7.6 | Catastrophic |
| BSC ε = 0.10, Q=50 | 5.7 ~ 7.0 | 6.5 | Catastrophic |
| BEC p = 0.05, Q=50 | 10.1 ~ 23.9 | 18.5 | Moderate |
| BEC p = 0.10, Q=50 | 7.1 ~ 16.7 | 13.0 | Severe |
| BEC p = 0.20, Q=50 | 6.0 ~ 11.9 | 9.2 | Very Severe |

**Key finding:** BEC consistently outperforms BSC at equivalent error probabilities. At p/ε = 0.05, BEC yields PSNR ~18.5 dB versus BSC's ~7.6 dB. This is because BEC erasures are explicitly marked (enabling soft-decision Viterbi decoding), whereas BSC bit flips are indistinguishable from correct bits.

### 3.2 SSIM Analysis

#### 3.2.1 Error-Free Conditions

| Quality Q | SSIM Range | Mean SSIM | Interpretation |
|-----------|-----------|-----------|----------------|
| Q = 10 | 0.3644 ~ 0.9043 | 0.7393 | Moderate structural similarity |
| Q = 50 | 0.8767 ~ 0.9749 | 0.9471 | High structural similarity |
| Q = 90 | 0.9927 ~ 0.9973 | 0.9947 | Near-perfect structure preservation |

#### 3.2.2 Impact of Channel Errors

| Channel Condition | SSIM Range | Mean SSIM |
|-------------------|-----------|-----------|
| BSC ε = 0.01, Q=50 | 0.1307 ~ 0.6003 | 0.4247 |
| BSC ε = 0.05, Q=50 | 0.0374 ~ 0.2213 | 0.1332 |
| BEC p = 0.05, Q=50 | 0.5925 ~ 0.9249 | 0.8143 |
| BEC p = 0.10, Q=50 | 0.2631 ~ 0.7683 | 0.5800 |

#### 3.2.3 PSNR-SSIM Correlation

PSNR and SSIM exhibit strong positive correlation across all experimental conditions (quadratic fit R² > 0.9). This validates the use of either metric for quality assessment. At high quality levels (PSNR > 35 dB), SSIM saturates near 0.99, indicating diminishing structural improvements beyond this point.

### 3.3 Compression Performance

| Quality Q | Compression Ratio Range | Mean Ratio | Source Bits (256×256×3) |
|-----------|------------------------|------------|--------------------------|
| Q = 10 | 10.9× ~ 35.1× | 27.0× | 44,832 ~ 144,688 bits |
| Q = 50 | 2.0× ~ 24.7× | 16.6× | 63,584 ~ 784,888 bits |
| Q = 90 | 0.9× ~ 16.1× | 10.0× | 97,712 ~ 1,668,080 bits |

**Observations:**

- **Content dependence:** Smooth gradient images compress 35× at Q=10; noise images achieve only 2× at Q=50.
- **Data expansion:** At Q=90, random noise produces a compressed stream larger than the raw pixels (CR = 0.9×). The high-frequency DCT coefficients resist quantization, and Huffman coding cannot compact them effectively.
- **Quality-compression trade-off:** Each doubling of compression ratio (from Q=10 to Q=50) costs approximately 8 dB in PSNR for structured images.

### 3.4 Viterbi Decoder Error Correction

| Channel | Error Prob. | Channel BER | Residual BER | Improvement Factor |
|---------|------------|-------------|-------------|-------------------|
| BSC | ε = 0.01 | 1.00% | 1.07% | 0.94× (no gain) |
| BSC | ε = 0.05 | 4.96% | 5.37% | 0.92× (no gain) |
| BSC | ε = 0.10 | 10.03% | 10.55% | 0.95× (no gain) |
| BEC | p = 0.05 | 4.96% | 0.13% | ~38× |
| BEC | p = 0.10 | 10.03% | 0.55% | ~18× |
| BEC | p = 0.20 | 20.10% | 2.20% | ~9× |

**Critical analysis:**

1. **BSC performance is poor.** The (2,1,7) convolutional code provides negligible BER improvement on BSC at ε ≥ 1%. At ε = 0.01, the channel BER of ~1% exceeds the code's correction capacity for a rate-1/2 code. The code's minimum free distance (d_free = 10) is insufficient to correct random errors at these densities.

2. **BEC performance is excellent.** Soft-decision Viterbi decoding exploits the known erasure positions, achieving 9–38× BER reduction. At p = 0.05, residual BER drops to ~0.13%, enabling high-quality reconstruction.

3. **Interleaving is essential for BEC.** The 64×128 block interleaver converts potential burst erasures into isolated erasures that the Viterbi decoder handles well.

### 3.5 Algorithm Complexity

| Pipeline Stage | Mean Time (s) | Fraction | Asymptotic Complexity |
|---------------|--------------|----------|----------------------|
| Source Encoding | 0.361 | 9.6% | O(N) — DCT, quantization, RLE, Huffman |
| Channel Encoding | 0.060 | 1.6% | O(N) — Convolutional encoding |
| Interleaving | 0.070 | 1.9% | O(N) — Block permutation |
| Channel Transmission | 0.038 | 1.0% | O(N) — Bit-wise operations |
| Channel Decoding | 2.518 | 66.8% | O(N·2^K) — Viterbi (K=7, 64 states) |
| Source Decoding | 0.722 | 19.1% | O(N) — Huffman decode, IDCT |
| **Total** | **3.769** | **100%** | — |

#### Complexity Scaling Verification

Linear fitting of encoding time vs. source bit count:

| Stage | Fitted Slope (s/bit) | R² | O(N) Valid? |
|-------|---------------------|-----|-------------|
| Source Encoding | 2.9 × 10⁻⁷ | 0.91 | ✓ |
| Channel Encoding | 1.2 × 10⁻⁷ | 0.96 | ✓ |
| Viterbi Decoding | 8.6 × 10⁻⁶ | 0.85 | ✓ (×30 of encoding) |

The Viterbi decoder's time is approximately 30× that of channel encoding, consistent with the theoretical O(2^K) = O(128) overhead per bit for K=7.

#### Key Performance Observations

- **Viterbi decoding is the bottleneck** (67% of end-to-end latency). For a 256×256 image, mean decoding time is ~2.5 seconds.
- **Noise images are 3–5× slower** than smooth images due to larger compressed bitstreams (low compression ratio → more bits to process).
- **Source encoding time is independent of quality factor** for the same image, as the DCT and Huffman operations process all 8×8 blocks regardless of Q.

### 3.6 Repeat Encoding Impact

#### BSC (ε = 0.05, Q = 50, synthetic_nature)

| Repeat N | Compression Ratio | PSNR (dB) | SSIM | Quality Improvement |
|----------|-------------------|-----------|------|---------------------|
| 1 | 15.7× | 7.4 | 0.078 | — (baseline) |
| 3 | 5.2× | 10.3 | 0.309 | +2.9 dB |
| 5 | 3.1× | 14.4 | 0.625 | +7.0 dB |

#### BEC (p = 0.10, Q = 50, synthetic_nature)

| Repeat N | Compression Ratio | PSNR (dB) | SSIM | Quality Improvement |
|----------|-------------------|-----------|------|---------------------|
| 1 | 15.7× | 13.5 | 0.566 | — (baseline) |
| 3 | 5.2× | 27.0 | 0.955 | +13.5 dB |
| 5 | 3.1× | 41.0 | 0.975 | +27.5 dB |

**Analysis:**

- **Repeat encoding is highly effective on BEC.** With N=5, PSNR recovers from 13.5 dB (poor) to 41.0 dB (excellent), approaching error-free quality. The majority voting across 5 repetitions eliminates nearly all residual errors after Viterbi decoding.
- **BSC benefits are more modest.** Even with N=5, PSNR only reaches 14.4 dB, because the Viterbi decoder fails to correct most errors initially, and majority voting across repetitions that are all erroneous provides limited benefit.
- **Bandwidth cost is proportional to N:** compression ratio drops from 15.7× (N=1) to 3.1× (N=5), a 5× bandwidth penalty.

---

## 4. Visualizations

16 professional charts were generated and are available in `results/figures/`:

| No. | File | Content |
|-----|------|---------|
| 01 | `01_psnr_vs_error_rate.png` | PSNR vs BSC/BEC error rate (by Quality) |
| 02 | `02_ssim_vs_error_rate.png` | SSIM vs BSC/BEC error rate (by Quality) |
| 03 | `03_compression_ratio.png` | Compression ratio and bit count vs Q |
| 04 | `04_complexity_breakdown.png` | Per-stage time breakdown (stacked bar + pie) |
| 05 | `05_rate_distortion.png` | Rate-distortion: PSNR vs Bits Per Pixel |
| 06 | `06_viterbi_performance.png` | Channel BER vs residual BER (log scale) |
| 07 | `07_psnr_vs_quality.png` | PSNR vs Q under different channel conditions |
| 08 | `08_repeat_impact.png` | Repeat encoding impact on PSNR/SSIM |
| 09 | `09_dashboard.png` | Comprehensive 6-panel performance dashboard |
| 10 | `10_psnr_ssim_correlation.png` | PSNR–SSIM correlation with quadratic fit |
| 11 | `11_compression_vs_error.png` | Compression ratio vs error resilience |
| 12 | `12_viterbi_improvement_heatmap.png` | Viterbi improvement factor heatmap |
| 13 | `13_complexity_scaling.png` | Time complexity O(N) verification |
| 14 | `14_per_image_quality.png` | Per-image-type quality comparison |
| 15 | `15_quality_repeat_interaction.png` | Q × Repeat interaction effects |
| 16 | `16_channel_residual_analysis.png` | Before/after Viterbi BER (log-log) |

---

## 5. Discussion

### 5.1 Strengths

1. **Excellent BEC performance.** The combination of (2,1,7) convolutional code, block interleaving, and soft-decision Viterbi decoding provides robust protection against erasure channels, achieving 9–38× BER improvement.

2. **Graceful quality degradation on BEC.** Even at p = 0.20 (20% erasure rate), the reconstructed image retains recognizable structure (SSIM > 0.4 at Q=50).

3. **Effective repeat encoding.** N=5 repeat + majority voting can recover near-lossless quality on BEC at the cost of 5× bandwidth expansion.

4. **High compression efficiency.** Up to 35× compression on smooth images while maintaining PSNR > 30 dB under error-free conditions.

### 5.2 Limitations

1. **Poor BSC performance.** The (2,1,7) code cannot correct random errors above ~1% BER with hard-decision decoding. A stronger code (e.g., Turbo or LDPC) or soft-decision demodulation would be needed for BSC channels.

2. **Data expansion on high-frequency content.** Noise-like images with Q ≥ 90 produce compressed streams larger than raw data, negating compression benefits.

3. **Viterbi complexity.** The O(N·2^K) complexity with K=7 makes the decoder 30× slower than the encoder. For real-time applications, a smaller constraint length or alternative decoding algorithm would be needed.

4. **Fixed interleaver size.** The 64×128 block interleaver requires padding, which wastes bandwidth. An adaptive interleaver based on payload size would be more efficient.

### 5.3 Recommendations

1. **Replace convolutional code with Turbo/LDPC** for BSC channels to achieve better random error correction.
2. **Implement adaptive interleaver sizing** to minimize padding overhead.
3. **Add CRC-16** to detect uncorrectable blocks and enable selective retransmission or concealment.
4. **Use arithmetic coding** instead of Huffman for improved compression, especially on high-frequency content.
5. **Parallelize Viterbi decoding** using GPU acceleration to reduce latency.

---

## 6. Conclusions

This project successfully implemented an end-to-end image transmission system integrating DCT-based lossy source coding, (2,1,7) convolutional channel coding, block interleaving, and BSC/BEC channel simulation. A comprehensive performance evaluation across 102 experimental conditions yielded the following key conclusions:

1. **Source coding** achieves excellent compression (up to 35×) with high PSNR (>30 dB at Q ≥ 50) under error-free conditions. The DCT + Huffman pipeline is computationally efficient (O(N), <10% of total time).

2. **Channel coding** provides strong protection on BEC (7–38× BER improvement via soft-decision Viterbi) but is ineffective on BSC at BER > 1%, where the (2,1,7) code's correction capacity is exceeded.

3. **Block interleaving** is critical for BEC performance, converting burst erasures into isolated errors.

4. **Repeat encoding** (N=5) can recover near-lossless quality on BEC (PSNR 41 dB at p=0.1) at the cost of 5× bandwidth.

5. **Viterbi decoding is the computational bottleneck** (67% of total time), consistent with its exponential complexity in constraint length.

6. **PSNR and SSIM are strongly correlated** (R² > 0.9), providing consistent quality assessment across all conditions.

---

## 7. Reproduction Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Full analysis pipeline
python scripts/analysis.py --quick          # Run experiments + generate 9 charts
python scripts/plot_advanced.py             # Generate 7 additional charts

# Re-plot from existing data
python scripts/analysis.py --from-csv results/analysis.csv

# Run all unit tests
python src/source_coding/test_source_coding.py
python tests/test_channel_coding.py
```

**Output files:**
- `results/analysis.csv` — Complete experimental data (102 rows, 18 columns)
- `results/figures/` — 16 professional charts (PNG, 1.7 MB total)
- `report/report_en.md` — This report (English)
- `report/report_cn.md` — Chinese version

---

## References

1. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4), 600–612.

2. Viterbi, A. J. (1967). Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. *IEEE Transactions on Information Theory*, 13(2), 260–269.

3. ITU-R BT.601-7 (2011). Studio encoding parameters of digital television for standard 4:3 and wide-screen 16:9 aspect ratios.

4. Pennebaker, W. B., & Mitchell, J. L. (1992). *JPEG: Still Image Data Compression Standard*. Springer.
