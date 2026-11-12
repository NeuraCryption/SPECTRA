import unittest
import torch
from spectra.compressor import SpectraCompressor

class TestSpectraQuantizer(unittest.TestCase):
    def test_group32_enforcement(self):
        with self.assertRaises(ValueError):
            SpectraCompressor(group_size=64)
        with self.assertRaises(ValueError):
            SpectraCompressor(group_size=16)
        # 32 succeeds
        c = SpectraCompressor(group_size=32)
        self.assertEqual(c.group_size, 32)

    def test_hadamard_disabled(self):
        with self.assertRaises(NotImplementedError):
            SpectraCompressor(hadamard_incoherence=True)

    def test_constant_nonzero_group_quantization(self):
        # When an entire group has identical nonzero weights
        w_pos = torch.full((1, 32), 2.5, dtype=torch.float32)
        packed, scales, zeros = SpectraCompressor._quantize_chunk_group32(w_pos, group_size=32)

        # Dequantize
        low = (packed & 0x0F).float()
        high = ((packed >> 4) & 0x0F).float()
        unpacked_q = torch.cat([low.reshape(1, 1, 16), high.reshape(1, 1, 16)], dim=-1)
        z = zeros & 0x0F
        deq = (unpacked_q - z.float()) * scales.float().unsqueeze(-1)
        err = torch.max(torch.abs(w_pos - deq.reshape(1, 32))).item()
        self.assertAlmostEqual(err, 0.0, places=4)

    def test_constant_negative_group_quantization(self):
        w_neg = torch.full((1, 32), -4.25, dtype=torch.float32)
        packed, scales, zeros = SpectraCompressor._quantize_chunk_group32(w_neg, group_size=32)

        low = (packed & 0x0F).float()
        high = ((packed >> 4) & 0x0F).float()
        unpacked_q = torch.cat([low.reshape(1, 1, 16), high.reshape(1, 1, 16)], dim=-1)
        z = zeros & 0x0F
        deq = (unpacked_q - z.float()) * scales.float().unsqueeze(-1)
        err = torch.max(torch.abs(w_neg - deq.reshape(1, 32))).item()
        self.assertAlmostEqual(err, 0.0, places=4)

    def test_constant_zero_group_quantization(self):
        w_zero = torch.zeros((1, 32), dtype=torch.float32)
        packed, scales, zeros = SpectraCompressor._quantize_chunk_group32(w_zero, group_size=32)

        low = (packed & 0x0F).float()
        high = ((packed >> 4) & 0x0F).float()
        unpacked_q = torch.cat([low.reshape(1, 1, 16), high.reshape(1, 1, 16)], dim=-1)
        z = zeros & 0x0F
        deq = (unpacked_q - z.float()) * scales.float().unsqueeze(-1)
        err = torch.max(torch.abs(w_zero - deq.reshape(1, 32))).item()
        self.assertAlmostEqual(err, 0.0, places=4)

if __name__ == "__main__":
    unittest.main()
