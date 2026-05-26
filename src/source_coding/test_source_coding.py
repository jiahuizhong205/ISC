"""
Unit tests for the source coding module (encoder + decoder).

Covers: DCT reversibility, quantisation, zigzag/RLE round-trip,
full encode→decode pipeline, and quality-vs-compression trade-off.
"""

import unittest

import numpy as np

from src.source_coding.encoder import (
    JPEGEncoder,
    _Huffman,
    _build_quant_table,
    _bytes_to_ints,
    _dct2d,
    _ints_to_bytes,
    _inverse_zigzag,
    _pad_to_multiple,
    _QY,
    _QC,
    _rle_decode,
    _rle_encode,
    _zigzag,
)
from src.source_coding.decoder import (
    JPEGDecoder,
    _HuffmanDecoder,
    _idct2d,
)


# ————————————————————————— helpers —————————————————————————


def _psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB).  Higher = better fidelity."""
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))


def _bpp(bits: list[int], h: int, w: int) -> float:
    """Bits per pixel from the compressed bitstream length."""
    return len(bits) / (h * w * 3)


# ————————————————————————— test cases —————————————————————————


class TestDCTReversibility(unittest.TestCase):
    """DCT → IDCT should be identity up to floating-point precision."""

    def test_random_blocks(self):
        for _ in range(20):
            block = np.random.randn(8, 8).astype(np.float64)
            rec = _idct2d(_dct2d(block))
            self.assertLess(np.abs(block - rec).max(), 1e-12)

    def test_constant_block(self):
        block = np.full((8, 8), 127.0, dtype=np.float64)
        rec = _idct2d(_dct2d(block))
        self.assertLess(np.abs(block - rec).max(), 1e-12)

    def test_identity_energy(self):
        """Orthonormal DCT preserves Frobenius norm."""
        block = np.random.randn(8, 8).astype(np.float64)
        dct_block = _dct2d(block)
        self.assertAlmostEqual(
            np.linalg.norm(block, 'fro'),
            np.linalg.norm(dct_block, 'fro'),
            delta=1e-12,
        )


class TestQuantisation(unittest.TestCase):
    """Quantisation-table scaling and round-trip behaviour."""

    def test_quality_50_uses_base_table(self):
        qy = _build_quant_table(_QY, 50)
        np.testing.assert_array_equal(qy, _QY)

    def test_quality_1_maximum_compression(self):
        qy = _build_quant_table(_QY, 1)
        # Q=1 scales table by 50×, values clipped at 255
        self.assertTrue(np.all(qy >= _QY))

    def test_quality_100_minimum_compression(self):
        qy = _build_quant_table(_QY, 100)
        self.assertTrue(np.all(qy <= 1.0 + 1e-9))

    def test_clamp_lower_bound(self):
        qy = _build_quant_table(_QY, 100)
        self.assertTrue(np.all(qy >= 1.0))

    def test_quantise_dequantise(self):
        """Quantise → dequantise round-trips through int rounding."""
        qy = _build_quant_table(_QY, 75)
        block = np.random.randn(8, 8) * 200
        quantised = np.round(block / qy)
        restored = quantised * qy
        diff = np.abs(block - restored).max()
        # Error should be bounded by 0.5 * max quant step
        self.assertLessEqual(diff, 0.5 * qy.max() + 1.0)


class TestZigzagRLE(unittest.TestCase):
    """Zigzag scan and run-length encoding round-trips."""

    def test_zigzag_inverse_roundtrip(self):
        block = np.arange(64, dtype=np.float64).reshape(8, 8)
        np.testing.assert_array_equal(block, _inverse_zigzag(_zigzag(block)))

    def test_rle_typical_block(self):
        coeffs = np.zeros(64, dtype=np.float64)
        coeffs[0] = 15.0
        coeffs[1] = 3.0
        coeffs[6] = -2.0
        encoded = _rle_encode(coeffs)
        decoded, nxt = _rle_decode(encoded, 0)
        np.testing.assert_array_almost_equal(coeffs, decoded)
        self.assertEqual(nxt, len(encoded))

    def test_rle_last_ac_nonzero(self):
        """No EOB emitted when index-63 AC is non-zero."""
        coeffs = np.zeros(64, dtype=np.float64)
        coeffs[0] = 5.0
        coeffs[63] = 7.0
        encoded = _rle_encode(coeffs)
        decoded, nxt = _rle_decode(encoded, 0)
        np.testing.assert_array_almost_equal(coeffs, decoded)
        self.assertEqual(nxt, len(encoded))

    def test_rle_all_zero_ac(self):
        coeffs = np.zeros(64, dtype=np.float64)
        coeffs[0] = 42.0
        encoded = _rle_encode(coeffs)
        decoded, nxt = _rle_decode(encoded, 0)
        np.testing.assert_array_almost_equal(coeffs, decoded)
        self.assertEqual(encoded, [42, 0, 0])  # dc + EOB

    def test_rle_long_zero_run(self):
        """Single spike at zigzag position 63."""
        coeffs = np.zeros(64, dtype=np.float64)
        coeffs[0] = 10.0
        coeffs[63] = -5.0
        encoded = _rle_encode(coeffs)
        decoded, nxt = _rle_decode(encoded, 0)
        np.testing.assert_array_almost_equal(coeffs, decoded)

    def test_rle_multiple_blocks_in_sequence(self):
        """Decoder can parse consecutive blocks from a flat list."""
        coeffs_a = np.zeros(64); coeffs_a[0] = 10; coeffs_a[1] = 2
        coeffs_b = np.zeros(64); coeffs_b[0] = 20; coeffs_b[3] = -1
        flat = _rle_encode(coeffs_a) + _rle_encode(coeffs_b)

        dec_a, idx = _rle_decode(flat, 0)
        dec_b, idx = _rle_decode(flat, idx)
        np.testing.assert_array_almost_equal(coeffs_a, dec_a)
        np.testing.assert_array_almost_equal(coeffs_b, dec_b)
        self.assertEqual(idx, len(flat))


class TestHuffman(unittest.TestCase):
    """Huffman encode → decode round-trip."""

    def test_roundtrip(self):
        data = b'hello world' * 20
        huff = _Huffman()
        huff.build(data)
        bits = huff.encode(data)
        dec = _HuffmanDecoder(huff.table)
        self.assertEqual(data, dec.decode(bits))

    def test_single_symbol(self):
        huff = _Huffman()
        huff.build(b'\x00' * 100)
        bits = huff.encode(b'\x00\x00')
        self.assertEqual(bits, [0, 0])
        dec = _HuffmanDecoder(huff.table)
        self.assertEqual(b'\x00\x00', dec.decode(bits))

    def test_empty_data(self):
        huff = _Huffman()
        huff.build(b'')
        self.assertEqual(huff.codes, {})

    def test_all_bits_are_zero_or_one(self):
        data = bytes(range(256)) * 10
        huff = _Huffman()
        huff.build(data)
        bits = huff.encode(data)
        self.assertTrue(all(b in (0, 1) for b in bits))


class TestIntsBytes(unittest.TestCase):
    """Signed 16-bit int ↔ bytes conversion."""

    def test_roundtrip(self):
        values = [0, 1, -1, 127, -128, 300, -300, 32767, -32768]
        self.assertEqual(values, _bytes_to_ints(_ints_to_bytes(values)))

    def test_empty(self):
        self.assertEqual([], _bytes_to_ints(_ints_to_bytes([])))

    def test_output_length(self):
        self.assertEqual(len(_ints_to_bytes([1, 2, 3])), 6)


class TestPadding(unittest.TestCase):
    """Symmetric padding to multiples of 8."""

    def test_already_aligned(self):
        img = np.ones((16, 24, 3), dtype=np.float64)
        padded, oh, ow = _pad_to_multiple(img)
        self.assertEqual((oh, ow), (16, 24))
        self.assertEqual(padded.shape, img.shape)

    def test_odd_dimensions(self):
        img = np.ones((15, 17, 3), dtype=np.float64)
        padded, oh, ow = _pad_to_multiple(img)
        self.assertEqual((oh, ow), (15, 17))
        self.assertEqual(padded.shape, (16, 24, 3))

    def test_single_channel(self):
        img = np.ones((10, 10), dtype=np.float64)
        padded, oh, ow = _pad_to_multiple(img)
        self.assertEqual(padded.shape, (16, 16))


class TestEncodeDecodePipeline(unittest.TestCase):
    """End-to-end encode → decode without channel errors."""

    @classmethod
    def setUpClass(cls):
        cls.encoder = JPEGEncoder(quality=75)
        cls.decoder = JPEGDecoder()

    def _roundtrip(self, image: np.ndarray):
        encoded = self.encoder.encode(image)
        decoded = self.decoder.decode(encoded['bits'], encoded['header'])
        return encoded, decoded

    # —— shape preservation ——

    def test_exact_8x8(self):
        img = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        _, rec = self._roundtrip(img)
        self.assertEqual(rec.shape, img.shape)

    def test_multiple_of_8(self):
        img = np.random.randint(0, 256, (32, 24, 3), dtype=np.uint8)
        _, rec = self._roundtrip(img)
        self.assertEqual(rec.shape, img.shape)

    def test_odd_dimensions(self):
        img = np.random.randint(0, 256, (15, 17, 3), dtype=np.uint8)
        _, rec = self._roundtrip(img)
        self.assertEqual(rec.shape, (15, 17, 3))

    def test_non_square(self):
        img = np.random.randint(0, 256, (8, 32, 3), dtype=np.uint8)
        _, rec = self._roundtrip(img)
        self.assertEqual(rec.shape, img.shape)

    # —— fidelity ——

    def test_solid_gray_perfect(self):
        """Uniform block: DC only → lossless after quantisation (Q75)."""
        img = np.full((16, 16, 3), 128, dtype=np.uint8)
        _, rec = self._roundtrip(img)
        np.testing.assert_array_equal(img, rec)

    def test_smooth_gradient_high_psnr(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        for i in range(32):
            img[i, :, 0] = (i * 8) % 256
            img[:, i, 1] = (i * 8) % 256
        _, rec = self._roundtrip(img)
        self.assertGreater(_psnr(img, rec), 30)

    def test_naturalistic_acceptable(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        _, rec = self._roundtrip(img)
        self.assertGreater(_psnr(img, rec), 20)

    # —— bitstream properties ——

    def test_bits_are_binary(self):
        img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        encoded, _ = self._roundtrip(img)
        self.assertTrue(all(b in (0, 1) for b in encoded['bits']))

    def test_header_contains_required_keys(self):
        img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        encoded, _ = self._roundtrip(img)
        for key in ('orig_h', 'orig_w', 'quality', 'huffman_table'):
            self.assertIn(key, encoded['header'])

    def test_huffman_table_roundtrips(self):
        """The Huffman table in the header must be usable by the decoder."""
        img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        encoded, _ = self._roundtrip(img)
        table = encoded['header']['huffman_table']
        self.assertIsInstance(table, dict)
        self.assertTrue(len(table) > 0)
        # Every code must be a non-empty bit string
        for code in table.values():
            self.assertIsInstance(code, str)
            self.assertTrue(all(c in '01' for c in code))


class TestQualityCompressionTradeoff(unittest.TestCase):
    """PSNR and bitrate across quality factors 10, 50, 90."""

    IMG = np.random.RandomState(42).randint(0, 256, (64, 48, 3)).astype(np.uint8)

    def _measure(self, quality: int):
        enc = JPEGEncoder(quality=quality)
        dec = JPEGDecoder()
        encoded = enc.encode(self.IMG)
        reconstructed = dec.decode(encoded['bits'], encoded['header'])
        bpp_val = _bpp(encoded['bits'], *self.IMG.shape[:2])
        psnr_val = _psnr(self.IMG, reconstructed)
        return bpp_val, psnr_val, len(encoded['bits']), encoded['header']

    def test_q10_high_compression_low_psnr(self):
        bpp_val, psnr_val, bits_len, _ = self._measure(10)
        self.assertLess(bpp_val, 1.0, f'Q10 should compress heavily, got {bpp_val:.3f} bpp')
        self.assertGreater(psnr_val, 10, f'Q10 PSNR too low: {psnr_val:.1f} dB')

    def test_q90_low_compression_high_psnr(self):
        bpp_val, psnr_val, bits_len, _ = self._measure(90)
        self.assertGreater(psnr_val, 28, f'Q90 PSNR too low: {psnr_val:.1f} dB')

    def test_quality_monotonic(self):
        """Higher quality → more bits AND higher PSNR."""
        bpp_10, psnr_10, _, _ = self._measure(10)
        bpp_50, psnr_50, _, _ = self._measure(50)
        bpp_90, psnr_90, _, _ = self._measure(90)

        self.assertLess(bpp_10, bpp_90, 'Q10 should use fewer bits than Q90')
        self.assertLess(psnr_10, psnr_90, 'Q10 PSNR should be lower than Q90')

        # Q50 should sit between Q10 and Q90
        self.assertLess(bpp_10, bpp_50)
        self.assertLess(bpp_50, bpp_90)
        self.assertLess(psnr_10, psnr_50)
        self.assertLess(psnr_50, psnr_90)

    def test_q_extremes_produce_valid_output(self):
        for q in (1, 100):
            enc = JPEGEncoder(quality=q)
            dec = JPEGDecoder()
            encoded = enc.encode(self.IMG)
            rec = dec.decode(encoded['bits'], encoded['header'])
            self.assertEqual(rec.shape, self.IMG.shape)
            self.assertEqual(rec.dtype, np.uint8)


class TestDCdifferential(unittest.TestCase):
    """DC differential coding compresses smooth images better."""

    def test_smooth_vs_random_bitrate(self):
        rng = np.random.RandomState(0)
        smooth = np.zeros((32, 32, 3), dtype=np.uint8)
        for i in range(32):
            smooth[i, :, :] = i * 8

        noisy = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        enc = JPEGEncoder(quality=75)
        dec = JPEGDecoder()

        enc_s = enc.encode(smooth)
        enc_n = enc.encode(noisy)

        rec_s = dec.decode(enc_s['bits'], enc_s['header'])
        rec_n = dec.decode(enc_n['bits'], enc_n['header'])

        self.assertLess(len(enc_s['bits']), len(enc_n['bits']),
                        'smooth image should compress better than noise')

        self.assertGreater(_psnr(smooth, rec_s), _psnr(noisy, rec_n),
                           'smooth image should reconstruct better')


if __name__ == '__main__':
    unittest.main()
