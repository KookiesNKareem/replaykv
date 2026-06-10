#!/usr/bin/env python3
"""Convert a torch tags table (replaykv/learned_tags.py output) to .npz."""
import argparse

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
args = ap.parse_args()
payload = torch.load(args.inp, map_location="cpu", weights_only=True)
np.savez_compressed(args.out, topm=payload["topm"].numpy().astype(np.int32),
                    k=np.int64(payload["k"]), vocab=np.int64(payload["vocab"]))
print(f"converted {args.inp} -> {args.out} (k={int(payload['k'])}, vocab={int(payload['vocab'])})")
