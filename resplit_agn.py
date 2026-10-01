"""Cut the AGN HDF5 file to a redshift range and re-split it by AGN parameter set.

Every AGN parameter set was simulated at both 2 h and 8 h. Splitting by file can put the
2 h copy in validation and the 8 h copy in test, so the two sets share underlying AGN.
This script groups spectra by parameter set (filename without the exposure time) and
assigns whole groups to validation or test, so each AGN appears in only one split.
The split is stratified by redshift, so validation and test cover the same redshifts.

Usage:
    uv run python resplit_agn.py all_agn_float32_v4.h5 all_agn_float32_v4_z0.9-1.7.h5
    uv run python resplit_agn.py IN.h5 OUT.h5 --zmin 0.9 --zmax 1.7 --test-size 0.2 --seed 42
"""
import argparse
import os
import re
import sys

import h5py
import numpy as np

SPLITS = ("validation", "test")


def group_key(obj_id):
    """AGN parameter set: filename without exposure time and processing suffix.
    AGN_temp_z0.9_ebv0.2_L300044.0_emline0.5_fragal0.0_2h_noisy_deZ_rebinned.fits
      -> AGN_temp_z0.9_ebv0.2_L300044.0_emline0.5_fragal0.0"""
    key, n = re.subn(r"_\d+h_noisy_deZ_rebinned\.fits$", "", obj_id)
    if n != 1:
        raise ValueError(f"unexpected AGN filename format: {obj_id}")
    return key


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("in_h5")
    p.add_argument("out_h5")
    p.add_argument("--zmin", type=float, default=0.9)
    p.add_argument("--zmax", type=float, default=1.7)
    p.add_argument("--test-size", type=float, default=0.2, help="fraction of parameter sets in test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.out_h5) and not args.overwrite:
        sys.exit(f"{args.out_h5} exists (use --overwrite)")
    if os.path.abspath(args.out_h5) == os.path.abspath(args.in_h5):
        sys.exit("output must be a different file from the input")

    # ---- read both existing splits and pool them ----
    with h5py.File(args.in_h5, "r") as hf:
        root_attrs = dict(hf.attrs)
        row_keys = [k for k in hf[SPLITS[0]] if k != "skipped_norm_con"]
        data = {k: np.concatenate([hf[s][k][:] for s in SPLITS]) for k in row_keys}
        skipped = np.concatenate([hf[s]["skipped_norm_con"][:] for s in SPLITS
                                  if "skipped_norm_con" in hf[s]])
        str_dtypes = {k: hf[SPLITS[0]][k].dtype for k in row_keys}

    ids = np.array([b.decode() for b in data["obj_id"]])
    z = data["redshift"].astype(float)
    n_total = len(ids)

    # ---- redshift cut (inclusive at both ends) ----
    keep = (z >= args.zmin - 1e-6) & (z <= args.zmax + 1e-6)
    print(f"redshift cut {args.zmin} <= z <= {args.zmax}: kept {keep.sum()} of {n_total} spectra")
    print("  redshifts removed:", sorted({float(v) for v in np.round(z[~keep], 3)}))
    data = {k: v[keep] for k, v in data.items()}
    ids, z = ids[keep], z[keep]

    # ---- group by parameter set and split whole groups ----
    groups = np.array([group_key(i) for i in ids])
    unique_groups = np.unique(groups)
    sizes = np.unique(np.unique(groups, return_counts=True)[1], return_counts=True)
    print(f"{len(unique_groups)} parameter sets; spectra per set: "
          + ", ".join(f"{s}: {c} sets" for s, c in zip(*sizes)))

    # split whole parameter sets, stratified by redshift so both splits cover the same z values
    rng = np.random.default_rng(args.seed)
    group_z = {g: zz for g, zz in zip(groups, np.round(z, 3))}
    test_groups = set()
    for zval in sorted(set(group_z.values())):
        at_z = rng.permutation(sorted(g for g, zz in group_z.items() if zz == zval))
        test_groups.update(at_z[:int(round(args.test_size * len(at_z)))])
    is_test = np.array([g in test_groups for g in groups])
    assignment = {"validation": ~is_test, "test": is_test}

    # no parameter set may appear in both splits
    assert not (set(groups[assignment["validation"]]) & set(groups[assignment["test"]]))

    # ---- write ----
    with h5py.File(args.out_h5, "w") as out:
        for k, v in root_attrs.items():
            out.attrs[k] = v
        out.attrs["source_file"] = os.path.basename(args.in_h5)
        out.attrs["z_range"] = [args.zmin, args.zmax]
        out.attrs["split_method"] = "by AGN parameter set (exposure times kept together)"
        out.attrs["split_seed"] = args.seed
        out.attrs["test_fraction_of_sets"] = args.test_size
        # spectra dropped earlier by save_h5 (continuum <= 0), pooled from both old splits
        out.create_dataset("skipped_norm_con", data=skipped)

        for split, mask in assignment.items():
            g = out.create_group(split)
            for k in row_keys:
                g.create_dataset(k, data=data[k][mask], dtype=str_dtypes[k])
            exp = data["EXPTIME"][mask] if "EXPTIME" in data else None
            exp_txt = (", ".join(f"{e:g}h: {c}" for e, c in zip(*np.unique(exp, return_counts=True)))
                       if exp is not None else "n/a")
            zs = z[mask]
            print(f"{split:<11} {mask.sum():5d} spectra, {len(set(groups[mask])):5d} parameter sets, "
                  f"z {zs.min():.2f}-{zs.max():.2f}, exposure {exp_txt}")

    print(f"wrote {args.out_h5}")


if __name__ == "__main__":
    main()
