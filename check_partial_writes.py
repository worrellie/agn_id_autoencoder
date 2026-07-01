import os
from astropy.io import fits

PROC_DIR = "processed_spectra"     # flat output dir
EXPECTED_PIXELS = 1549
SUFFIX = "_deZ_rebinned.fits"

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
            for k in ("OG_Z", "SNR"):
                if k not in hdr:
                    return f"missing header key {k}"
    except Exception as e:
        return f"unreadable: {type(e).__name__}"
    return None

def main():
    print(f"scanning {PROC_DIR}/ ...", flush=True)

    bad = []
    checked = 0
    with os.scandir(PROC_DIR) as it:        # streams entries, no big list built
        for entry in it:
            if not entry.name.endswith(SUFFIX):
                continue
            why = check_file(entry.path)
            if why:
                bad.append((entry.path, why))
            checked += 1
            if checked % 5000 == 0:
                print(f"  {checked} checked, {len(bad)} bad so far", flush=True)

    print(f"\n{checked} files checked, {len(bad)} bad", flush=True)
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