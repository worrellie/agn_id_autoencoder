import h5py, numpy as np

with h5py.File("all_spectra_float32_v2.h5", "r") as hf:
    for split in ("train", "validation", "test"):
        g = hf[split]
        raw = g["raw_flux"][:2000]
        log = g["log_scale_flux"][:2000]
        raw_nan, log_nan = np.isnan(raw), np.isnan(log)

        same     = np.array_equal(raw_nan, log_nan)      # gap structure consistent?
        log_only = int((log_nan & ~raw_nan).sum())       # NaN at a DATA pixel = real bug
        n_inf    = int(np.isinf(log).sum())              # inf anywhere = real bug
        per_row  = log_nan.sum(1)

        print(f"{split}: masks_identical={same}  log_only_NaN={log_only}  inf={n_inf}  "
              f"gaps/row min={per_row.min()} max={per_row.max()} mean={per_row.mean():.0f}")