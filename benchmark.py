#!/usr/bin/env python3
"""
Lean training benchmark: reuses your real H5SpecDataset + autoencoder + loss,
stripped of wandb / plotting / analysis. Produces comparable numbers across
preload modes (none/ram/vram) and across machines (laptop / cluster / vast.ai).

Examples:
  python benchmark.py --data all_spectra_float32.h5 --preload none --device cuda
  python benchmark.py --data all_spectra_float32.h5 --preload ram  --device cuda
  python benchmark.py --data all_spectra_float32.h5 --preload vram --device cuda

  # just re-print the comparison table for everything logged, no run:
  python benchmark.py --summary

Each run appends one JSON line to --out and then prints a side-by-side table of
all modes logged for the current config, so it fills in as you run each mode.
"""

import argparse, json, os, platform, resource, socket, statistics, time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import optim

# --- your real code (same modules run_ae.py uses) --------------------------- #
from datahandling import H5SpecDataset, make_dataloader
import autoencoder as ae

_LAYER_CONFIGS = {
    1: [{"in": 512, "out": 256}],
    2: [{"in": 512, "out": 256}, {"in": 256, "out": 64}],
    3: [{"in": 512, "out": 256}, {"in": 256, "out": 128}, {"in": 128, "out": 64}],
    4: [{"in": 700, "out": 512}, {"in": 512, "out": 256},
        {"in": 256, "out": 128}, {"in": 128, "out": 64}],
}
MODE_ORDER = {"none": 0, "ram": 1, "vram": 2}


# ---- your real loss, verbatim from training.py ----------------------------- #
def _loss_calc_batch(x_hat, x, x_mask, mu=None, logvar=None, beta=0):
    batch_size = x_hat.shape[0]
    n_unmasked_pixels = x_mask.sum(dim=1)
    sq_err_per_element = (x_hat - x) ** 2
    masked_sq_err = sq_err_per_element * x_mask
    masked_mse_per_sample = masked_sq_err.sum(dim=1) / n_unmasked_pixels
    mean_masked_mse_for_batch = masked_mse_per_sample.sum() / batch_size
    if mu is not None and logvar is not None:
        kl_divs = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        mean_kl_div_for_batch = kl_divs.sum() / batch_size
    else:
        mean_kl_div_for_batch = torch.tensor(0.0).to(x.device)
    return mean_masked_mse_for_batch + (beta * mean_kl_div_for_batch)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def build_model(args, input_size):
    """model normalize=False: standardisation happens in the dataset now
    (and the model's forward ignores the flag anyway)."""
    config = _LAYER_CONFIGS[args.n_layers]
    cls = ae.StandardAutoencoder if args.model_type == "standard" else ae.VAEAutoencoder
    return cls(config, input_size, args.latent, args.flux_type,
               args.model_normalize, activation=args.activation)


def compute_loss(model, batch, mask, args):
    x_hat, mu, logvar = model(batch)
    return _loss_calc_batch(x_hat, batch, mask.float(),
                            mu=mu, logvar=logvar, beta=args.beta)


def meta(device, args, input_size, n_params):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "device": device,
        "device_name": (torch.cuda.get_device_name(0) if device == "cuda" else "cpu"),
        "preload": args.preload,
        "model_type": args.model_type,
        "n_layers": args.n_layers,
        "latent": args.latent,
        "activation": args.activation,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "input_size": input_size,
        "n_params": n_params,
    }


def load_rows(out_path):
    if not os.path.exists(out_path):
        return []
    rows = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def print_comparison(rows, match=None):
    """Print modes side by side. If `match` given, filter to rows sharing its
    config (so only none/ram/vram of the current config show)."""
    if match is not None:
        keys = ("model_type", "n_layers", "latent", "batch_size", "device", "input_size")
        rows = [r for r in rows if all(r.get(k) == match.get(k) for k in keys)]
    if not rows:
        return
    # if comparing one config, keep only the most recent row per mode
    if match is not None:
        latest = {}
        for r in rows:
            latest[r.get("preload")] = r      # later lines overwrite earlier
        rows = list(latest.values())
    rows = sorted(rows, key=lambda r: MODE_ORDER.get(r.get("preload"), 9))

    if match is not None:
        m = match
        print(f"\ncomparison  [{m['model_type']} nl{m['n_layers']} ls{m['latent']} "
              f"bs{m['batch_size']} on {m['device']}]")
    else:
        print("\nall logged runs")
    hdr = (f"{'mode':<6}{'step(ms)':>10}{'samp/s':>10}{'epoch(s)':>10}"
           f"{'150ep(m)':>10}{'GPU(MB)':>9}{'RAM(MB)':>9}{'speedup':>9}")
    print(hdr)
    print("-" * len(hdr))
    base = None
    for r in rows:
        sps = r.get("samples_per_s") or 0
        if base is None:
            base = sps
        spd = f"{sps/base:.2f}x" if base else "-"
        gpu = r.get("peak_gpu_mem_mb")
        gpu_s = f"{gpu:,.0f}" if gpu is not None else "-"
        print(f"{r.get('preload',''):<6}"
              f"{r.get('median_step_ms',0):>10.2f}"
              f"{sps:>10,.0f}"
              f"{r.get('est_epoch_s',0):>10.1f}"
              f"{r.get('est_full_run_min',0):>10.1f}"
              f"{gpu_s:>9}"
              f"{r.get('peak_host_rss_mb',0):>9,.0f}"
              f"{spd:>9}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data")
    p.add_argument("--split", default="train")
    p.add_argument("--flux-type", default="log_scale_flux", dest="flux_type")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--preload", default="ram", choices=["none", "ram", "vram"])
    p.add_argument("--model-type", default="standard", dest="model_type",
                   choices=["standard", "vae"])
    p.add_argument("--n-layers", type=int, default=1, choices=[1, 2, 3, 4], dest="n_layers")
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--activation", default="LeakyReLU", choices=["ReLU", "Tanh", "LeakyReLU"])
    p.add_argument("--model-normalize", action="store_true", dest="model_normalize")
    p.add_argument("--no-standardize", action="store_false", dest="standardize")
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--epochs-est", type=int, default=150, dest="epochs_est")
    p.add_argument("--out", default="benchmark_results.jsonl")
    p.add_argument("--summary", action="store_true",
                   help="just print the table of all logged runs and exit")
    args = p.parse_args()

    # summary-only mode: print everything logged and exit
    if args.summary:
        print_comparison(load_rows(args.out), match=None)
        return
    if not args.data:
        raise SystemExit("--data is required (or use --summary to just print the table)")

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.preload == "vram" and device != "cuda":
        raise SystemExit("preload=vram requires --device cuda")

    ds = H5SpecDataset(args.data, split=args.split, flux_type=args.flux_type,
                       standardize=args.standardize, preload=args.preload, device=device)
    loader = make_dataloader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

    input_size = ds[0][0].shape[0]
    model = build_model(args, input_size).to(device)
    n_params = sum(x.numel() for x in model.parameters())
    opt = optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    per_step, seen, step = [], 0, 0
    data_iter = iter(loader)
    while step < args.warmup + args.steps:
        try:
            batch, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch, mask = next(data_iter)

        nb = (args.preload != "vram")
        batch = batch.to(device, non_blocking=nb)
        mask = mask.to(device, non_blocking=nb)
        batch = torch.nan_to_num(batch, nan=0.0)

        if step == 0:
            v = batch[mask.bool()]
            print(f"[sanity] valid-pixel mean={v.mean().item():.3f} "
                  f"std={v.std().item():.3f}  (expect ~0.0 / ~1.0 if standardized)")

        sync(device)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = compute_loss(model, batch, mask, args)
        loss.backward()
        opt.step()
        sync(device)
        dt = time.perf_counter() - t0

        if step >= args.warmup:
            per_step.append(dt)
            seen += batch.shape[0]
        step += 1

    final_loss = loss.item()          # .item() detaches -> no requires_grad warning
    if not np.isfinite(final_loss):
        print("WARNING: loss is not finite -- check NaN scrubbing / data")

    total = sum(per_step)
    steps_per_epoch = len(ds) // args.batch_size
    epoch_s = statistics.median(per_step) * steps_per_epoch
    peak_gpu = (torch.cuda.max_memory_allocated() / 1024**2) if device == "cuda" else None
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    result = {
        **meta(device, args, input_size, n_params),
        "median_step_ms": statistics.median(per_step) * 1e3,
        "samples_per_s": seen / total,
        "steps_per_epoch": steps_per_epoch,
        "est_epoch_s": epoch_s,
        "est_full_run_min": epoch_s * args.epochs_est / 60,
        "peak_gpu_mem_mb": peak_gpu,
        "peak_host_rss_mb": peak_rss,
        "final_loss": final_loss,
    }
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")

    # single-run summary
    print("=" * 60)
    print(f"mode={args.preload}  device={device}  {args.model_type} nl{args.n_layers} ls{args.latent}  bs{args.batch_size}")
    print(f"params           : {n_params:,}")
    print(f"median step      : {result['median_step_ms']:.2f} ms")
    print(f"throughput       : {result['samples_per_s']:,.0f} samples/s")
    if peak_gpu is not None:
        print(f"peak GPU mem     : {peak_gpu:,.0f} MB")
    print(f"peak host RAM    : {peak_rss:,.0f} MB")
    print(f"est. epoch       : {epoch_s:.1f} s")
    print(f"est. {args.epochs_est} epochs   : {result['est_full_run_min']:.1f} min")
    print(f"final loss       : {final_loss:.4f}")
    print("=" * 60)

    # side-by-side comparison of all modes logged for THIS config
    print_comparison(load_rows(args.out), match=result)


if __name__ == "__main__":
    main()