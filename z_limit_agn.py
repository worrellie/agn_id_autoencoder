import h5py
import numpy as np


def _copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def filter_h5_by_redshift(input_h5, output_h5, z_min=0.9, z_max=1.7):
    """
    Copy input_h5 to output_h5, keeping only sources with
    z_min <= redshift <= z_max in every group that has a 'redshift' dataset.
    Everything else (root datasets, groups without redshift) is copied unchanged.
    """
    with h5py.File(input_h5, "r") as hf_in, h5py.File(output_h5, "w") as hf_out:
        _copy_attrs(hf_in, hf_out)

        for name, obj in hf_in.items():
            if not isinstance(obj, h5py.Group) or "redshift" not in obj:
                print(f"Copying '{name}' unchanged (no redshift to filter on).")
                hf_in.copy(obj, hf_out, name=name)
                continue

            z = obj["redshift"][()].ravel()
            n_src = z.shape[0]
            mask = (z >= z_min) & (z <= z_max)
            print(f"Split '{name}': keeping {mask.sum()} of {n_src} sources "
                  f"in z = [{z_min}, {z_max}]")

            grp_out = hf_out.create_group(name)
            _copy_attrs(obj, grp_out)

            for dname, dset in obj.items():
                if not isinstance(dset, h5py.Dataset):
                    print(f"  '{name}/{dname}' is a group; copied unfiltered.")
                    obj.copy(dset, grp_out, name=dname)
                    continue

                data = dset[()]
                if data.ndim > 0 and data.shape[0] == n_src:
                    data = data[mask]
                else:
                    print(f"  '{name}/{dname}' shape {data.shape} doesn't match "
                          f"{n_src} sources; copied unfiltered.")

                # Chunking/compression only if the source had it and the result is non-empty
                kwargs = {}
                if dset.chunks is not None and data.size > 0:
                    kwargs = dict(
                        chunks=tuple(min(c, s) for c, s in zip(dset.chunks, data.shape)),
                        compression=dset.compression,
                        compression_opts=dset.compression_opts,
                        shuffle=dset.shuffle,
                        fletcher32=dset.fletcher32,
                    )

                dset_out = grp_out.create_dataset(dname, data=data, **kwargs)
                _copy_attrs(dset, dset_out)

    print(f"\nCreated filtered HDF5: {output_h5}")