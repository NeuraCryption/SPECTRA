import unittest
import torch
from spectra.compressor import SpectraCompressor
from spectra.kernels import fused_w4a16_gemv_hdsq, batched_w4a16_gemm

class TestSpectraKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available for Triton kernel testing")

    def _prepare_quantized_weights(self, N=128, K=256, num_outliers=0):
        torch.manual_seed(42)
        w = torch.randn(N, K, dtype=torch.float16, device="cuda")
        outlier_idx = None
        outlier_w = None
        if num_outliers > 0:
            outlier_idx = torch.randperm(K, device="cuda")[:num_outliers]
            outlier_w = w[:, outlier_idx].clone()

        packed_w, scales, zeros = SpectraCompressor._compute_group32_quant(w, outlier_idx=outlier_idx)
        packed_w = packed_w.cuda()
        scales = scales.cuda()

        pz = zeros.cuda()
        z_c = torch.empty((N, pz.shape[1] * 2), dtype=torch.uint8, device="cuda")
        z_c[:, 0::2] = pz & 0x0F
        z_c[:, 1::2] = (pz >> 4) & 0x0F
        z_c = z_c[:, :K // 32]

        low = (packed_w & 0x0F).float()
        high = ((packed_w >> 4) & 0x0F).float()
        low_grp = low.reshape(N, K // 32, 16)
        high_grp = high.reshape(N, K // 32, 16)
        unpacked_q = torch.cat([low_grp, high_grp], dim=-1)
        w_deq = ((unpacked_q - z_c.unsqueeze(-1).float()) * scales.unsqueeze(-1).float()).reshape(N, K).to(torch.float16)

        if num_outliers > 0:
            w_deq[:, outlier_idx] = outlier_w

        return w_deq, packed_w, scales, z_c, outlier_idx, outlier_w

    def test_gemv_vs_ref(self):
        w_deq, packed_w, scales, z_c, _, _ = self._prepare_quantized_weights(N=128, K=256)
        x = torch.randn(1, 256, dtype=torch.float16, device="cuda")
        ref_out = torch.matmul(x, w_deq.t())
        kernel_out = fused_w4a16_gemv_hdsq(x, packed_w, scales, z_c, marlin_layout=True)
        max_diff = torch.max(torch.abs(ref_out - kernel_out)).item()
        self.assertLessEqual(max_diff, 0.05)

    def test_batched_gemm_vs_ref(self):
        w_deq, packed_w, scales, z_c, _, _ = self._prepare_quantized_weights(N=128, K=256)
        for batch_size in (1, 2, 4, 8):
            x = torch.randn(batch_size, 256, dtype=torch.float16, device="cuda")
            ref_out = torch.matmul(x, w_deq.t())
            kernel_out = batched_w4a16_gemm(x, packed_w, scales, z_c, marlin_layout=True)
            max_diff = torch.max(torch.abs(ref_out - kernel_out)).item()
            self.assertLessEqual(max_diff, 0.05, f"Diff too large for batch size {batch_size}")

    def test_batched_gemm_with_hdsq_outliers(self):
        w_deq, packed_w, scales, z_c, outlier_idx, outlier_w = self._prepare_quantized_weights(N=128, K=256, num_outliers=16)
        x = torch.randn(4, 256, dtype=torch.float16, device="cuda")
        ref_out = torch.matmul(x, w_deq.t())
        kernel_out = batched_w4a16_gemm(
            x, packed_w, scales, z_c,
            outlier_idx=outlier_idx, outlier_w=outlier_w,
            marlin_layout=True
        )
        max_diff = torch.max(torch.abs(ref_out - kernel_out)).item()
        self.assertLessEqual(max_diff, 0.05)

if __name__ == "__main__":
    unittest.main()
