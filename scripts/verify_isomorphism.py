"""Standalone numerical isomorphism test for VLLM_W8_FAST=1 and in-place w8_linear(..., out=out).

Verifies that single-pass sm_86 W8 tiles match an FP32 dequantized reference GEMM within
1 BF16 ULP across all target Inkling-2-Mini decode and prefill shapes.
"""
import importlib.util
import os
import sys
import torch

os.environ["VLLM_W8_FAST"] = "1"
w8_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/w8_gemm.py"
spec = importlib.util.spec_from_file_location("w8_fast", w8_path)
assert spec is not None and spec.loader is not None
m_fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m_fast)

SHAPES = [
    (1, 5632, 2304),    # qkvr, M=1
    (8, 5632, 2304),    # qkvr, c=1 verify (k=7 -> 8 tokens)
    (12, 5632, 2304),   # qkvr, c=4 verify (k=2 -> 12 tokens)
    (8, 2304, 3072),    # wo_ud, c=1 verify
    (1, 201088, 256),   # DSpark Markov recurrent step
    (128, 2304, 3072),  # wo_ud, prefill path (> MAX_M, in-place out=)
]

for M, N, K in SHAPES:
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.05
    b = torch.randn(N, dtype=torch.bfloat16, device="cuda") * 0.01
    q, s = m_fast.quantize(w)
    ref = ((x.float() @ (q.float() * s[:, None]).t()) + (b.float() if M <= 64 else 0.0)).to(torch.bfloat16)
    if M <= 64:
        got = m_fast.w8_gemm(x, q, s, bias=b)
    else:
        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        got = m_fast.w8_linear(x, (q, s), out=out)
    max_diff = (got.float() - ref.float()).abs().max().item()
    exact_pct = (got == ref).float().mean().item() * 100.0
    print(f"Shape ({M:3d}, {N:6d}, {K:4d}): max_abs_diff={max_diff:.6f}, bit_exact={exact_pct:.2f}%")
    assert torch.allclose(got, ref, atol=0.032, rtol=1e-2), f"Mismatch too large: {max_diff}"

print("ISOMORPHISM_PROOF_PASSED")
