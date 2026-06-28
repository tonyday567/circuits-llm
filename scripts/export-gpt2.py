#!/usr/bin/env python3
"""Export GPT-2 weights from HuggingFace to raw float32 files.

Usage:
    python3 scripts/export-gpt2.py [--size small|medium|large|xl] [--out weights/]

Outputs a directory of .f32 files readable by Circuit.LLM.Weights.loadGpt2.
"""

import argparse
import numpy as np
import struct
import os
import sys

GPT2_SIZES = {
    "small":  "gpt2",
    "medium": "gpt2-medium",
    "large":  "gpt2-large",
    "xl":     "gpt2-xl",
}

GPT2_CONFIG = {
    "small":  {"vocab": 50257, "n_embd": 768,  "n_head": 12, "n_layer": 12},
    "medium": {"vocab": 50257, "n_embd": 1024, "n_head": 16, "n_layer": 24},
    "large":  {"vocab": 50257, "n_embd": 1280, "n_head": 20, "n_layer": 36},
    "xl":     {"vocab": 50257, "n_embd": 1600, "n_head": 25, "n_layer": 48},
}


def write_f32(path: str, arr: np.ndarray):
    """Write a numpy array as raw little-endian float32."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = np.asarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    print(f"  wrote {path}  shape={list(arr.shape)}  size={arr.nbytes}")


def export_gpt2(model_name: str, out_dir: str, cfg: dict):
    """Download GPT-2 weights and export to raw .f32 files."""
    from transformers import GPT2Model
    import torch

    print(f"Loading {model_name} from HuggingFace...")
    model = GPT2Model.from_pretrained(model_name)
    sd = model.state_dict()
    n_embd = cfg["n_embd"]
    n_layer = cfg["n_layer"]

    # Token embeddings
    write_f32(f"{out_dir}/wte.f32", sd["wte.weight"].numpy())        # [vocab, n_embd]
    write_f32(f"{out_dir}/wpe.f32", sd["wpe.weight"].numpy())        # [1024, n_embd]

    for h in range(n_layer):
        pfx = f"h.{h}."
        # Layer norm 1
        write_f32(f"{out_dir}/h{h}.ln1.gamma.f32", sd[f"{pfx}ln_1.weight"].numpy())
        write_f32(f"{out_dir}/h{h}.ln1.beta.f32",  sd[f"{pfx}ln_1.bias"].numpy())

        # Fused QKV attention weight [n_embd, 3*n_embd]
        qkv_w = sd[f"{pfx}attn.c_attn.weight"].numpy()  # [n_embd, 3*n_embd] (PyTorch transposed convention)
        # HuggingFace stores as [n_embd, 3*n_embd] which is W^T in PyTorch convention
        # Our code expects [n_embd, 3*n_embd] as well (row-major, columns are output dims)
        write_f32(f"{out_dir}/h{h}.attn.qkv.w.f32", qkv_w)
        write_f32(f"{out_dir}/h{h}.attn.qkv.b.f32", sd[f"{pfx}attn.c_attn.bias"].numpy())

        # Output projection
        proj_w = sd[f"{pfx}attn.c_proj.weight"].numpy()  # [n_embd, n_embd]
        write_f32(f"{out_dir}/h{h}.attn.proj.w.f32", proj_w)
        write_f32(f"{out_dir}/h{h}.attn.proj.b.f32", sd[f"{pfx}attn.c_proj.bias"].numpy())

        # Layer norm 2
        write_f32(f"{out_dir}/h{h}.ln2.gamma.f32", sd[f"{pfx}ln_2.weight"].numpy())
        write_f32(f"{out_dir}/h{h}.ln2.beta.f32",  sd[f"{pfx}ln_2.bias"].numpy())

        # MLP fc (first linear) [n_embd, 4*n_embd]
        fc_w = sd[f"{pfx}mlp.c_fc.weight"].numpy()
        write_f32(f"{out_dir}/h{h}.mlp.fc.w.f32", fc_w)
        write_f32(f"{out_dir}/h{h}.mlp.fc.b.f32", sd[f"{pfx}mlp.c_fc.bias"].numpy())

        # MLP proj (second linear) [4*n_embd, n_embd]
        proj2_w = sd[f"{pfx}mlp.c_proj.weight"].numpy()
        write_f32(f"{out_dir}/h{h}.mlp.proj.w.f32", proj2_w)
        write_f32(f"{out_dir}/h{h}.mlp.proj.b.f32", sd[f"{pfx}mlp.c_proj.bias"].numpy())

    # Final layer norm
    write_f32(f"{out_dir}/lnf.gamma.f32", sd["ln_f.weight"].numpy())
    write_f32(f"{out_dir}/lnf.beta.f32",  sd["ln_f.bias"].numpy())

    print(f"\nDone. Weights exported to {out_dir}/")
    print(f"Load in Haskell with: loadGpt2 \"{out_dir}\" Gpt2Small")


def export_synthetic(out_dir: str, cfg: dict):
    """Export synthetic random weights for testing (no HuggingFace download needed)."""
    n_embd = cfg["n_embd"]
    n_layer = cfg["n_layer"]
    vocab = cfg["vocab"]
    rng = np.random.RandomState(42)

    def r(*shape):
        return rng.randn(*shape).astype(np.float32) * 0.02

    write_f32(f"{out_dir}/wte.f32", r(vocab, n_embd))
    write_f32(f"{out_dir}/wpe.f32", r(1024, n_embd))

    for h in range(n_layer):
        write_f32(f"{out_dir}/h{h}.ln1.gamma.f32", r(n_embd))
        write_f32(f"{out_dir}/h{h}.ln1.beta.f32",  np.zeros(n_embd, dtype=np.float32))
        write_f32(f"{out_dir}/h{h}.attn.qkv.w.f32", r(n_embd, 3 * n_embd))
        write_f32(f"{out_dir}/h{h}.attn.qkv.b.f32", r(3 * n_embd))
        write_f32(f"{out_dir}/h{h}.attn.proj.w.f32", r(n_embd, n_embd))
        write_f32(f"{out_dir}/h{h}.attn.proj.b.f32", r(n_embd))
        write_f32(f"{out_dir}/h{h}.ln2.gamma.f32", r(n_embd))
        write_f32(f"{out_dir}/h{h}.ln2.beta.f32",  np.zeros(n_embd, dtype=np.float32))
        write_f32(f"{out_dir}/h{h}.mlp.fc.w.f32", r(n_embd, 4 * n_embd))
        write_f32(f"{out_dir}/h{h}.mlp.fc.b.f32", r(4 * n_embd))
        write_f32(f"{out_dir}/h{h}.mlp.proj.w.f32", r(4 * n_embd, n_embd))
        write_f32(f"{out_dir}/h{h}.mlp.proj.b.f32", r(n_embd))

    write_f32(f"{out_dir}/lnf.gamma.f32", r(n_embd))
    write_f32(f"{out_dir}/lnf.beta.f32",  np.zeros(n_embd, dtype=np.float32))

    print(f"\nDone. Synthetic weights exported to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Export GPT-2 weights")
    parser.add_argument("--size", default="small", choices=["small", "medium", "large", "xl"],
                        help="Model size (default: small)")
    parser.add_argument("--out", default="weights/", help="Output directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic random weights instead of downloading")
    args = parser.parse_args()

    cfg = GPT2_CONFIG[args.size]

    if args.synthetic:
        cfg2 = {k: v for k, v in cfg.items()}
        # Use smaller dimensions for synthetic test
        cfg2["vocab"] = 1000
        cfg2["n_embd"] = 32
        cfg2["n_head"] = 4
        cfg2["n_layer"] = 2
        export_synthetic(args.out, cfg2)
    else:
        export_gpt2(GPT2_SIZES[args.size], args.out, cfg)


if __name__ == "__main__":
    main()
