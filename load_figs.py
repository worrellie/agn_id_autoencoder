import pickle as pkl
from matplotlib import pyplot as plt

load_path = "RUN_vae_nl4_ls32_e10_ReLU_B1e-03_lr1e-04_wd1e-08_esFalse\RUN_vae_nl4_ls32_e10_ReLU_B1e-03_lr1e-04_wd1e-08_esFalse_loss.pkl"

with open(load_path, 'rb') as p:
	fig = pkl.load(p)

plt.show()