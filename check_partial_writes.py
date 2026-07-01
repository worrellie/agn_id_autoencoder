import os, glob
from astropy.io import fits

PROC_DIR = "processed_spectra"     # the flat output dir
EXPECTED_PIXELS = 1549             # your known grid length

def check_file(path):
    """Return None if healthy, else a short reason string."""
    try:
        with fits.open(path) as hdul:
            if len(hdul) < 2 or hdul[1].data is None:
                return "no/empty table HDU"
            n = len(hdul[1].data["flux"])
            if n != EXPECTED_PIXELS:
                return f"wrong length: {n} (expected {EXPECTED_PIXELS})"
            hdr = hdul[1].header
            for k in ("OG_Z", "SNR"):          # keys the H5 reader needs
                if k not in hdr:
                    return f"missing header key {k}"
    except Exception as e:
        return f"unreadable: {type(e).__name__}"   # truncated/corrupt -> raises here
    return None

def main():
    files = glob.glob(os.path.join(PROC_DIR, "*_deZ_rebinned.fits"))
    print(f"checking {len(files)} files in {PROC_DIR}/ ...")

    bad = []
    for i, f in enumerate(files):
        why = check_file(f)
        if why:
            bad.append((f, why))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(files)}  ({len(bad)} bad so far)")

    print(f"\n{len(bad)} bad file(s) of {len(files)}")
    for f, why in bad:
        print(f"  {why:35s} {os.path.basename(f)}")

    if bad:
        with open("bad_outputs.txt", "w") as fh:
            fh.writelines(f + "\n" for f, _ in bad)
        print(f"\nwrote {len(bad)} paths to bad_outputs.txt")
    else:
        print("all outputs look complete.")

if __name__ == "__main__":
    main()