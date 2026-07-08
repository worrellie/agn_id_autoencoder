#!/usr/bin/env python3
"""
Diagnose whether your training is input-pipeline-bound or compute-bound,
and demonstrate the RAM-preload fix -- all locally, NO good GPU required.

It does three things:
  1. Inspects your HDF5 array (shape, dtype, chunking, compression, RAM size).
  2. Times the CURRENT streaming access pattern (mirrors your H5SpecDataset:
     one random-index row read per __getitem__), sweeping num_workers.
  3. Times a PRELOADED version (whole array in a RAM tensor, reads are free).

The ratio (preloaded samples/s) / (streaming samples/s) is your answer:
a big gap == pipeline-bound, and preloading is the fix.

Needs only: torch, h5py, numpy.  Example:
  python diagnose_pipeline.py --data all_spectra.h5 --split train \
      --flux-type log_scale_flux --batch-size 256 --workers 0,2,4
"""

import argparse
import time
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader


# --- mirrors your H5SpecDataset: lazy handle, one random row per read -------- #
class StreamingH5(Dataset):
    def __init__(self, path, split, flux_type):
        with h5py.File(path, "r") as hf:
            self.len = hf[split][flux_type].shape[0]
            self.n_pixels = hf[split][flux_type].shape[1]
        self.path, self.split, self.flux_type = path, split, flux_type
        self.hf = None  # opened lazily inside each worker (fork-safe)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.path, "r")
        sample = torch.from_numpy(self.hf[self.split][self.flux_type][idx]).float()
        return sample, (sample != 0)


# --- the fix: whole array read once into RAM, __getitem__ just indexes it ---- #
class PreloadedRAM(Dataset):
    def __init__(self, path, split, flux_type):
        with h5py.File(path, "r") as hf:
            arr = hf[split][flux_type][:]  # single big sequential read
        self.data = torch.from_numpy(np.ascontiguousarray(arr)).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        s = self.data[idx]
        return s, (s != 0)


def inspect(path, split, flux_type):
    with h5py.File(path, "r") as hf:
        dset = hf[split][flux_type]
        ram_gib = dset.size * dset.dtype.itemsize / 1024**3
        print("=" * 64)
        print(f"array            : {split}/{flux_type}")
        print(f"shape            : {dset.shape}")
        print(f"dtype            : {dset.dtype}")
        print(f"chunks           : {dset.chunks}")
        print(f"compression      : {dset.compression} (opts={dset.compression_opts})")
        print(f"in-RAM size      : {ram_gib:.2f} GiB  (need ~{ram_gib*2:.1f} GiB free to preload)")
        print("=" * 64)


def time_loader(ds, batch_size, workers, n_batches, pin, device):
    # to see how long it takes to go through the batches (from the dateset)
    # will compare one by one (streaming) vs dumping everything in RAM (preloadng)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        drop_last=True, pin_memory=pin,
        persistent_workers=(workers > 0),
    )
    it = iter(loader) # convert to iterator
    for _ in range(3):  # warmup (worker spin-up, first reads)
        try:
            next(it)
        except StopIteration:
            it = iter(loader); next(it)
    seen, b = 0, 0
    t0 = time.perf_counter()
    for x, _m in it:
        if device is not None:
            x = x.to(device, non_blocking=pin)
            if device == "cuda":
                torch.cuda.synchronize()
        seen += x.shape[0]
        b += 1
        if b >= n_batches:
            break
    return seen / (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--flux-type", default="log_scale_flux")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", default="0,2,4", help="comma list, e.g. 0,2,4")
    p.add_argument("--n-batches", type=int, default=60)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="cuda = also time the host->device copy")
    args = p.parse_args()

    worker_counts = [int(w) for w in args.workers.split(",")]
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    pin = device == "cuda"
    dev_arg = device if device == "cuda" else None
    print(f"[diagnose] device for transfer test: {device}  pin_memory={pin}\n")

    inspect(args.data, args.split, args.flux_type)

    print("\nSTREAMING (current pattern) -- samples/sec:")
    stream_best = 0.0
    for w in worker_counts:
        sps = time_loader(StreamingH5(args.data, args.split, args.flux_type),
                          args.batch_size, w, args.n_batches, pin, dev_arg)
        stream_best = max(stream_best, sps)
        print(f"  num_workers={w:<2d} : {sps:>10,.0f}")

    print("\nPRELOADED to RAM -- samples/sec:")
    t0 = time.perf_counter()
    pre = PreloadedRAM(args.data, args.split, args.flux_type)
    load_s = time.perf_counter() - t0
    print(f"  (one-time load: {load_s:.1f} s, {pre.data.element_size()*pre.data.nelement()/1024**3:.2f} GiB in RAM)")
    pre_sps = time_loader(pre, args.batch_size, 0, args.n_batches, pin, dev_arg)
    print(f"  num_workers=0  : {pre_sps:>10,.0f}")

    print("\n" + "=" * 64)
    ratio = pre_sps / stream_best if stream_best else float("inf")
    print(f"preloaded / best-streaming throughput ratio : {ratio:.1f}x")
    if ratio >= 3:
        print("VERDICT: pipeline-bound. Preload to RAM -- it removes the wall.")
    elif ratio >= 1.5:
        print("VERDICT: partly pipeline-bound. Preloading still worth it.")
    else:
        print("VERDICT: streaming keeps up; bottleneck is elsewhere (likely compute).")
    print("=" * 64)


if __name__ == "__main__":
    main()