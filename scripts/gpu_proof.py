#!/usr/bin/env python
"""Prove the installed torch build can run on this node's GPU: print versions, the compiled CUDA
architecture list, and run a real forward pass. Exits non-zero if the GPU cannot be used.
Run it on a compute node (srun/sbatch), never on the head node."""
import sys

import torch

print("python", sys.version.split()[0])
print("torch.__version__", torch.__version__, "| torch.version.cuda", torch.version.cuda)
print("torch.cuda.get_arch_list()", torch.cuda.get_arch_list())
if not torch.cuda.is_available():
    sys.exit("FAIL: torch.cuda.is_available() is False")
cap = torch.cuda.get_device_capability(0)
print("device", torch.cuda.get_device_name(0), "| capability", cap, "| visible GPUs", torch.cuda.device_count())
if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
    print(f"WARNING: sm_{cap[0]}{cap[1]} is not in the arch list; the forward pass below is the real test")
try:
    native = torch.cuda.is_bf16_supported(including_emulation=False)
except TypeError:
    native = None
print("is_bf16_supported()", torch.cuda.is_bf16_supported(), "| native (including_emulation=False)", native)
with torch.no_grad():
    for dt in (torch.float16, torch.float32):
        layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True).to("cuda", dt).eval()
        y = layer(torch.randn(4, 32, 256, device="cuda", dtype=dt))
        assert torch.isfinite(y).all(), f"non-finite output in {dt}"
        print(f"forward pass OK on cuda:0 dtype={dt} out_shape={tuple(y.shape)} mean_abs={float(y.float().abs().mean()):.4f}")
try:
    import transformers
    print("transformers", transformers.__version__)
except ImportError:
    print("transformers not installed")
print("PROOF OK")
