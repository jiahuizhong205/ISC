"""
单元测试：信道编码模块。

测试项目：
1. 卷积码编解码的基本功能
2. 无错和有错场景
3. BSC 和 BEC 信道适配
"""

import sys
import random
from src.channel_coding.convolutional import ConvCodec
from src.channel_coding.interleaver import BlockInterleaver, RandomInterleaver


def test_conv_basic():
    """测试卷积码编码/解码基本功能（无错）。"""
    print("\n[Test 1] 卷积码基本编解码 (无错)")
    print("-" * 50)

    codec = ConvCodec()

    # 测试数据
    test_bits = [1, 0, 1, 1, 0, 0, 1, 0] * 2  # 16 bits

    # 编码
    encoded = codec.encode(test_bits)
    print(f"原始比特数: {len(test_bits)}")
    print(f"编码后比特数: {len(encoded)}")
    print(f"码率: {len(test_bits) / len(encoded):.3f}")

    # 完美接收 (BSC, ε=0)
    decoded = codec.decode(encoded, channel_type='bsc')
    print(f"解码比特数: {len(decoded)}")

    # 验证
    errors = sum(1 for a, b in zip(test_bits, decoded) if a != b)
    print(f"✓ 解码误差: {errors} bits" if errors == 0 else f"✗ 解码误差: {errors} bits")

    return errors == 0


def test_bsc_with_errors():
    """测试 BSC 信道有错场景下的 Viterbi 纠错。"""
    print("\n[Test 2] BSC 信道有错纠正")
    print("-" * 50)

    codec = ConvCodec()
    test_bits = [random.randint(0, 1) for _ in range(32)]

    # 编码
    encoded = codec.encode(test_bits)
    print(f"原始: {len(test_bits)} bits → 编码: {len(encoded)} bits")

    # 模拟 BSC 错误 (ε=0.1, 约 10% 错误率)
    error_rate = 0.1
    received = []
    num_errors = 0
    for bit in encoded:
        if random.random() < error_rate:
            received.append(1 - bit)
            num_errors += 1
        else:
            received.append(bit)

    print(f"注入错误: {num_errors}/{len(encoded)} ({100*num_errors/len(encoded):.1f}%)")

    # 解码
    decoded = codec.decode(received, channel_type='bsc')

    # 验证
    errors_raw = sum(1 for a, r in zip(test_bits, received) if a != r)
    errors_decoded = sum(1 for a, b in zip(test_bits, decoded) if a != b)
    print(f"原始接收误差: {errors_raw}/{len(test_bits)} ({100*errors_raw/len(test_bits):.1f}%)")
    print(f"解码后误差: {errors_decoded}/{len(test_bits)} ({100*errors_decoded/len(test_bits):.1f}%)")

    improved = errors_raw >= errors_decoded
    print(f"{'✓' if improved else '✗'} 纠错效果: {'改善' if improved else '未改善'}")

    return improved


def test_bec_with_erasures():
    """测试 BEC 信道擦除场景。"""
    print("\n[Test 3] BEC 信道擦除处理")
    print("-" * 50)

    codec = ConvCodec()
    test_bits = [random.randint(0, 1) for _ in range(32)]

    # 编码
    encoded = codec.encode(test_bits)
    print(f"原始: {len(test_bits)} bits → 编码: {len(encoded)} bits")

    # 模拟 BEC 擦除 (p=0.15)
    erasure_prob = 0.15
    received = []
    num_erasures = 0
    for bit in encoded:
        if random.random() < erasure_prob:
            received.append(None)
            num_erasures += 1
        else:
            received.append(bit)

    print(f"注入擦除: {num_erasures}/{len(encoded)} ({100*num_erasures/len(encoded):.1f}%)")

    # 解码
    decoded = codec.decode(received, channel_type='bec')

    # 验证
    errors_decoded = sum(1 for a, b in zip(test_bits, decoded) if a != b)
    print(f"解码后误差: {errors_decoded}/{len(test_bits)}")
    print(f"✓ BEC 解码完成")

    return len(decoded) == len(test_bits)


def test_interleaver_block():
    """测试块交织器。"""
    print("\n[Test 4] 块交织器")
    print("-" * 50)

    interleaver = BlockInterleaver(rows=4, cols=8)
    test_bits = list(range(32))  # [0, 1, 2, ..., 31]

    # 交织
    interleaved = interleaver.interleave(test_bits)
    print(f"原始: {test_bits[:16]}...")
    print(f"交织: {interleaved[:16]}...")

    # 解交织
    deinterleaved = interleaver.deinterleave(interleaved)
    print(f"解交: {deinterleaved[:16]}...")

    # 验证
    match = test_bits == deinterleaved
    print(f"✓ 交织/解交验证: {'通过' if match else '失败'}")

    return match


def test_interleaver_random():
    """测试随机交织器。"""
    print("\n[Test 5] 随机交织器")
    print("-" * 50)

    interleaver = RandomInterleaver(size=32, seed=42)
    test_bits = [random.randint(0, 1) for _ in range(64)]

    # 交织
    interleaved = interleaver.interleave(test_bits)
    print(f"原始长度: {len(test_bits)}, 交织长度: {len(interleaved)}")

    # 解交织
    deinterleaved = interleaver.deinterleave(interleaved)

    # 验证
    match = test_bits == deinterleaved
    print(f"✓ 交织/解交验证: {'通过' if match else '失败'}")

    return match


def main():
    print("\n" + "=" * 50)
    print("  信道编码模块单元测试")
    print("=" * 50)

    results = []
    try:
        results.append(("基本编解码", test_conv_basic()))
        results.append(("BSC 纠错", test_bsc_with_errors()))
        results.append(("BEC 擦除", test_bec_with_erasures()))
        results.append(("块交织器", test_interleaver_block()))
        results.append(("随机交织器", test_interleaver_random()))
    except Exception as e:
        print(f"\n✗ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 总结
    print("\n" + "=" * 50)
    print("  测试总结")
    print("=" * 50)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name:20s} {status}")

    all_passed = all(passed for _, passed in results)
    print("\n" + ("✓ 所有测试通过!" if all_passed else "✗ 有测试失败"))
    print("=" * 50 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
