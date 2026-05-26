# 成员 C 实现验证指南

以下每条命令均为单行，可直接复制粘贴到终端运行。

---

## C-1: BSC 信道仿真

```
python -c "from src.channel_model.channel import BSCChannel; bsc = BSCChannel(epsilon=0.05, seed=42); bits = [0,1]*5000; rx, rate = bsc.transmit(bits); print(f'实际误码率: {rate:.4f} (期望约 0.05)'); assert 0.04 < rate < 0.06, '失败: 误码率越界'; print('C-1 通过')"
```

**通过标准**：实际误码率在 `0.05 ± 0.01` 范围内。

---

## C-2: BEC 信道仿真

```
python -c "from src.channel_model.channel import BECChannel; bec = BECChannel(erasure_prob=0.1, seed=42); bits = [0,1]*5000; rx, rate = bec.transmit(bits); erased = sum(1 for b in rx if b is None); print(f'删除位: {erased}/10000, 实际删除率: {rate:.4f} (期望约 0.10)'); assert 0.08 < rate < 0.12, '失败: 删除率越界'; print('C-2 通过')"
```

**通过标准**：实际删除率在 `0.10 ± 0.02` 范围内。

---

## C-3: 接口适配层

```
python -c "from src.interfaces import SourceCodec, ChannelCodec, Channel; print(f'SourceCodec: {SourceCodec}'); print(f'ChannelCodec: {ChannelCodec}'); print(f'Channel:     {Channel}'); print('C-3 通过 (三个接口定义正确)')"
```

**通过标准**：三个接口类可正常导入，无报错。

---

## C-4: 主流水线 main.py

先创建测试图：
```
python -c "import numpy as np; from PIL import Image; Image.fromarray(np.random.RandomState(42).randint(0,256,(64,64,3),dtype=np.uint8)).save('data/test.png')"
```

然后依次运行：
```
python -m src.main --image data/test.png --channel bsc --param 0.0 --output output/test_lossless.png
python -m src.main --image data/test.png --channel bsc --param 0.05 --output output/test_bsc.png
python -m src.main --image data/test.png --channel bec --param 0.1 --output output/test_bec.png
python -m src.main --image data/test.png --channel bsc --param 0.0 --save-bin output/test.bin
```

**通过标准**：所有命令运行不报错，输出 PSNR 值，生成对应的 png 和 bin 文件。

---

## C-5: 比特流打包/拆包

```
python -c "from src.bitstream import pack_bitstream, unpack_bitstream; bits = [1,0,1,1,0,0,1,0,1,1,1,0]; header = {'shape': [64,64,3], 'quality': 50}; packed = pack_bitstream(bits, header); bits2, hdr2, pad = unpack_bitstream(packed); assert bits == bits2, f'失败: {bits} vs {bits2}'; assert header == hdr2, '失败: header 不一致'; print(f'C-5 通过 ({len(bits)} bits -> {len(packed)} bytes, padding={pad})')"
```

**通过标准**：编码→解码后比特流和 header 完全一致。

---

## C-6: 批量实验脚本

```
python scripts/run_experiments.py --channels bsc --subset 1
```

**通过标准**：输出进度信息，最终打印 "实验完成: N 组"。

---

## C-7: 实验结果 CSV 导出

```
python -c "import csv; f = open('results/experiments.csv'); rows = list(csv.DictReader(f)); fields = ['image','channel','param','quality','psnr','time_source_enc','time_channel_enc','time_transmission','time_channel_dec','time_source_dec','compression_ratio','actual_error_rate']; assert all(fld in rows[0] for fld in fields), '缺少字段'; assert all(row['psnr'] not in (None,'') for row in rows), 'PSNR 缺失'; print(f'C-7 通过 ({len(rows)} 行, {len(fields)} 字段完整)')"
```

**通过标准**：CSV 包含全部 12 个字段，每行的 PSNR 不为空。

---

## C-8: 端到端集成测试

```
python -c "import subprocess, sys; r = subprocess.run([sys.executable, '-m', 'src.main', '--image', 'data/test.png', '--channel', 'bsc', '--param', '0.0', '--output', 'output/check.png'], capture_output=True, text=True); assert r.returncode == 0, '失败'; print('C-8 通过 (端到端流水线无崩溃)')"
```

**通过标准**：运行不崩溃。

---

## 一键全量验证

复制整段到终端（每行独立粘贴）：

```
python -c "from src.channel_model.channel import BSCChannel; bsc = BSCChannel(0.05, seed=42); rx, r = bsc.transmit([0,1]*5000); assert 0.04 < r < 0.06; print('[C-1] BSC 信道 通过')"
```
```
python -c "from src.channel_model.channel import BECChannel; bec = BECChannel(0.1, seed=42); rx, r = bec.transmit([0,1]*5000); assert 0.08 < r < 0.12; print('[C-2] BEC 信道 通过')"
```
```
python -c "from src.interfaces import SourceCodec, ChannelCodec, Channel; print('[C-3] 接口定义 通过')"
```
```
python -c "from src.bitstream import pack_bitstream, unpack_bitstream; bits = [1,0,1,1,0,0,1,0,1,1,1,0]; header = {'shape': [64,64,3]}; packed = pack_bitstream(bits, header); bits2, hdr2, _ = unpack_bitstream(packed); assert bits == bits2 and header == hdr2; print('[C-5] 比特流打包 通过')"
```
```
python scripts/run_experiments.py --channels bsc --subset 1
```
```
python -m src.main --image data/test.png --channel bsc --param 0.0 --output output/check.png
```

---

## 预期结果速查表

| 场景 | 预期 PSNR (降级模式) | 说明 |
|---|---|---|
| BSC 0% | inf dB | 无损，无比特翻转 |
| BSC 1% | ~25 dB | 轻微损伤 |
| BSC 5% | ~18 dB | 中等损伤 |
| BSC 10% | ~15 dB | 严重损伤 |
| BEC 5% | ~19 dB | 擦除位填 0 |
| BEC 10% | ~17 dB | 擦除位填 0 |
| BEC 20% | ~14 dB | 大量擦除 |

> 注：降级模式指 A/B 模块未就绪时的 raw pixel bits 直通。当 A 的源编码和 B 的信道编码就绪后，PSNR 值会不同。
