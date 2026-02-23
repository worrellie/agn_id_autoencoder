import os
import numpy as np
from astropy.io import fits

def generate_sine_data(num_samples, seq_len):
    x = np.linspace(0, 2 * np.pi, seq_len, dtype =  np.float32)
    data = []
    for _ in range(num_samples):
        # Random phase and slight amplitude shift
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.8, 1.2)
        sample = amp * np.sin(x + phase)
        data.append(sample)
        
    return np.array(data, dtype =  np.float32), x




def get_raw_data(spec_dir = "/home/worrellie/Documents/phd/autoencoder/test_gal"):
        
    fluxes = []
    i=0
    for spec in os.listdir(spec_dir):
        i= i+1
        spec_path = os.path.join(spec_dir, spec)
        try:
            with fits.open(spec_path) as hdul:

                data = hdul[1].data
                flux = data['flux']
                # print(flux)
                l = data['lambda']
                l = l
                flux = flux.astype(np.float32)
                # flux = torch.from_numpy(flux)
                fluxes.append(flux)
                # if i == 1:
                #     plt.figure()
                #     plt.plot(l,flux)
                #     # plt.show()
    

        except Exception as e:
            print(f"Error opening spectrum: {spec} ({e})")
    
    fluxes = np.array(fluxes, dtype = np.float32)
    l = np.array(l, dtype = np.float32)
    # print(type(fluxes))
    # print(type(fluxes[0]))

    return fluxes, l # returns list of ndarrays