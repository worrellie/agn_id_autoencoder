import torch
from  datahandling import H5SpecDataset

def test_preload_modes_agree(tiny_h5):
    a = H5SpecDataset(tiny_h5, "validation", standardize=True, preload="none")
    b = H5SpecDataset(tiny_h5, "validation", standardize=True, preload="ram")
    for i in (0, 1, 17):
        xa, ma = a[i]; xb, mb = b[i]
        assert torch.equal(ma, mb)
        assert torch.allclose(xa, xb, rtol=1e-6, atol=1e-7)