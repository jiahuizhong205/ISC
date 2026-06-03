# 配置文件说明

项目支持通过 `config.json` 文件配置运行参数，无需每次在命令行中指定。

---

## 使用方法

### 方式一：仅用配置文件

```bash
# 编辑 config.json，然后直接运行
python -m src.main
```

### 方式二：配置文件 + 命令行覆盖

```bash
# 以 config.json 为基础，命令行覆盖个别参数
python -m src.main --param 0.005 --output output/bsc_05pct.png
```

### 方式三：指定不同的配置文件

```bash
python -m src.main --config config_bec.json
```

---

## 配置文件格式 (`config.json`)

```json
{
    "image": "data/kodim01.png",
    "channel": "bsc",
    "param": 0.01,
    "quality": 75,
    "seed": 42,
    "output": "output/result.png",
    "save_bin": ""
}
```

---

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `image` | string | `data/kodim01.png` | 输入图像路径 |
| `channel` | string | `bsc` | 信道类型：`bsc` 或 `bec` |
| `param` | float | `0.0` | 信道参数（BSC 的 ε 或 BEC 的 p） |
| `quality` | int | `50` | 源编码质量 1~100，越高越清晰 |
| `repeat` | int | `5` | 重复编码次数 1/3/5（1=最快 5=最强） |
| `seed` | int | `42` | 随机种子，相同种子可复现结果 |
| `output` | string | `output/result.png` | 重建图像输出路径 |
| `save_bin` | string | `""` | 比特流保存路径，空字符串表示不保存 |

---

## 常用场景配置

### 无损测试

```json
{
    "image": "data/kodim01.png",
    "channel": "bsc",
    "param": 0.0,
    "quality": 75,
    "seed": 42,
    "output": "output/clean.png",
    "save_bin": ""
}
```

### BSC 1% 误码

```json
{
    "image": "data/kodim01.png",
    "channel": "bsc",
    "param": 0.01,
    "quality": 75,
    "seed": 42,
    "output": "output/bsc_1pct.png",
    "save_bin": ""
}
```

### BEC 5% 删除

```json
{
    "image": "data/kodim01.png",
    "channel": "bec",
    "param": 0.05,
    "quality": 75,
    "seed": 42,
    "output": "output/bec_5pct.png",
    "save_bin": ""
}
```

---

## 优先级规则

```
命令行参数  >  配置文件  >  程序默认值
```

例如：
- `config.json` 中 `quality: 75`，命令行 `--quality 30` → 实际使用 30
- `config.json` 中 `param: 0.01`，命令行未指定 → 实际使用 0.01
- 都不指定 → 使用程序默认值

---

## `repeat` 参数效果对比

| repeat | BSC 1% PSNR | 压缩率 | 耗时 |
|---|---|---|---|
| 1 | ~10 dB | 6.0x | ~17s |
| 3 | ~18 dB | 2.0x | ~42s |
| 5 | ~28 dB | 1.2x | ~66s |
