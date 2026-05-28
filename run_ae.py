import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import pathlib as path
import h5py

import funcs
from datahandling import H5SpecDataset
import autoencoder as ae
import training
import argparse
import time

import wandb

import logging

logger = logging.getLogger(__name__)


def main():

	wandb.login(key="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")

	###############
	###############
	TESTING = False
	verb = TESTING
	###############
	###############

	# parse args

	parser = argparse.ArgumentParser()

	# for parallelisation
	parser.add_argument("--task_id", type=int, default=None)
	#

	parser.set_defaults(
		activation="ReLU",
		architecture=[
			{
				"in": 256,
				"out": 64,
			},
		],
		model_type="StandardAutoencoder",
	)

	parser.add_argument("-f", "--filename", default="all_spectra.h5")
	parser.add_argument("-p", "--project_name", default="unspecified_project")
	parser.add_argument("-ft", "--flux_type", default="normalized_flux_cont")

	# note this is for standardizing before training
	parser.add_argument( "-n", "--normalize", action="store_true" )  # if -n is parsed, True is returned

	parser.add_argument("-e", "--epochs", default=10, type=int)
	parser.add_argument( "-s", "--early_stop", action="store_true" )  # if -s is parsed, True is returned
	parser.add_argument("-b", "--beta", default=0.0, type=float)
	parser.add_argument("--learn_rate", default=1e-5, type=float)
	parser.add_argument("-d", "--weight_decay", default=1e-8, type=float)
	parser.add_argument("-l", "--latent", default=32, type=int)

	hidden_layers = parser.add_mutually_exclusive_group()
	hidden_layers.add_argument(
		"--layers-1",
		dest="architecture",
		action="store_const",
		const=[
			{"in": 512, "out": 256,},
		],
	)
	hidden_layers.add_argument(
		"--layers-2",
		dest="architecture",
		action="store_const",
		const=[
			{"in": 512, "out": 256,},
			{"in": 256, "out": 64,},
		],
	)
	hidden_layers.add_argument(
		"--layers-3",
		dest="architecture",
		action="store_const",
		const=[
			{"in": 512, "out": 256,},
			{"in": 256, "out": 128,},
			{"in": 128, "out": 64,},
		],
	)
	hidden_layers.add_argument(
		"--layers-4",
		dest="architecture",
		action="store_const",
		const=[
			{
				"in": 700,
				"out": 512,
			},
			{
				"in": 512,
				"out": 256,
			},
			{
				"in": 256,
				"out": 128,
			},
			{
				"in": 128,
				"out": 64,
			},
		],
	)

	activation_funcs = parser.add_mutually_exclusive_group()
	activation_funcs.add_argument(
		"-r", "--relu", dest="activation", action="store_const", const="ReLU"
	)
	activation_funcs.add_argument(
		"-t", "--tanh", dest="activation", action="store_const", const="Tanh"
	)
	activation_funcs.add_argument(
		"--leaky", dest="activation", action="store_const", const="LeakyReLU"
	)

	model_type = parser.add_mutually_exclusive_group()
	model_type.add_argument(
		"--sae", dest="model_type", action="store_const", const="StandardAutoencoder"
	)
	model_type.add_argument(
		"--vae", dest="model_type", action="store_const", const="VariationalAutoencoder"
	)

	args = parser.parse_args()

	#####################################################################################################
	# for running multiple tests in cluster
	# if args.task_id is not None:
	# 	test_configs = [
	# 		{
	# 			"epochs": 100,
	# 			"latent": 32,
	# 			"learn_rate": 1e-4,
	# 			"weight_decay": 1e-4,
	# 			"beta": 0.0,
	# 			"model_type": "StandardAutoencoder",
	# 			"layers": "--layers-4",
	# 			"flux_type" : "log_scale_flux",
	# 			"normalize" : "normalize"
	# 		},
	# 		{
	# 			"epochs": 500,
	# 			"latent": 32,
	# 			"learn_rate": 1e-4,
	# 			"weight_decay": 1e-4,
	# 			"beta": 0.0,
	# 			"model_type": "StandardAutoencoder",
	# 			"layers": "--layers-4",
	# 			"flux_type" : "log_scale_flux",
	# 			"normalize" : "normalize"
	# 		},
	# 		{
	# 			"epochs": 1000,
	# 			"latent": 32,
	# 			"learn_rate": 1e-4,
	# 			"weight_decay": 1e-4,
	# 			"beta": 0.0,
	# 			"model_type": "StandardAutoencoder",
	# 			"layers": "--layers-4",
	# 			"flux_type" : "log_scale_flux",
	# 			"normalize" : "normalize"
	# 		}, 
	# 	]
	# 	c = test_configs[args.task_id]
	# 	args.flux_type = c["flux_type"]
	# 	args.normalize = c["normalize"]
	# 	args.epochs = c["epochs"]
	# 	args.latent = c["latent"]
	# 	args.learn_rate = c["learn_rate"]
	# 	args.weight_decay = c["weight_decay"]
	# 	args.beta = c["beta"]
	# 	args.model_type = c["model_type"]
	# 	# map layers string to the architecture const
	# 	layer_map = {
	# 		"--layers-1": [{"in": 512, "out": 256}],

	# 		"--layers-2": [{"in": 512, "out": 256}, 
	# 					   {"in": 256, "out": 64}],

	# 		"--layers-3": [{"in": 512, "out": 256},
	# 					   {"in": 256, "out": 128},
	# 					   {"in": 128, "out": 64},],

	# 		"--layers-4": [{"in": 700, "out": 512,},
	# 					   {"in": 512, "out": 256,},
	# 					   {"in": 256, "out": 128, },
	# 					   {"in": 128, "out": 64, },],
	# 	}
	# 	args.architecture = layer_map[c["layers"]]

	#####################################################################################################

	CONFIG = args.architecture
	LATENT_SIZE = args.latent
	ACTIVATION_FUNCTION = args.activation
	EPOCHS = args.epochs
	EARLY_STOPPING = args.early_stop
	BETA = (
		args.beta
	)  # kl weighting only used in VAE, automatically set as 0 for other models
	LEARNING_RATE = args.learn_rate
	WEIGHT_DECAY = args.weight_decay

	normalize = args.normalize
	flux_type = args.flux_type

	#####################################################################################################

	TEST_NAME = f"RUN_{args.model_type}_nl{len(CONFIG)}_ls{LATENT_SIZE}_e{EPOCHS}_{ACTIVATION_FUNCTION}_B{BETA:.0e}_lr{LEARNING_RATE:.0e}_wd{WEIGHT_DECAY}_es{EARLY_STOPPING}_n{normalize}"

	counter = 1
	base_name = TEST_NAME
	while os.path.isdir(TEST_NAME):
		TEST_NAME = f"{base_name}_{counter}"
		counter += 1

	# make test dir
	funcs.make_test_dir(TEST_NAME, test=TESTING)

	#####################################################################################################

	# set up logger
	log_path = os.path.join(TEST_NAME, "output.log")

	root_logger = logging.getLogger()
	root_logger.setLevel(logging.INFO)

	file_handler = logging.FileHandler(log_path)
	console_handler = logging.StreamHandler()

	formatter = logging.Formatter(
		"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
	)
	file_handler.setFormatter(formatter)
	console_handler.setFormatter(formatter)

	root_logger.addHandler(file_handler)
	root_logger.addHandler(console_handler)

	#####################################################################################################

	# get device
	logger.info(f"GPU available: {torch.cuda.is_available()}")

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	logger.info(f"Device type: {device.type}")
	if device.type == "cpu":
		try:
			num_threads = int(os.environ["SLURM_CPUS_PER_TASK"])
		except:
			num_threads = min(4, os.cpu_count())
			logger.info(f"cannot get SLURM_CPUS_PER_TASK, defaulting to {num_threads}")
		finally:
			torch.set_num_threads(num_threads)
			logger.info(f"num threads set to: {torch.get_num_threads()}")

	#####################################################################################################
	# Load data

	if TESTING:
		batch_size_train = 2
		batch_size_valid = 1
	else:
		batch_size_train = batch_size_valid = 64

	DATA = args.filename



	# default is normalised data
	train = H5SpecDataset(DATA, split="train", flux_type=flux_type)
	valid = H5SpecDataset(DATA, split="validation", flux_type=flux_type)

	num_workers = 0
	if device.type == "cpu":
		if os.environ.get("SLURM_CPUS_PER_TASK") is not None:
			num_workers = 8

	train_loader = torch.utils.data.DataLoader(
		train, batch_size=batch_size_train, shuffle=True, num_workers=num_workers
	)
	valid_loader = torch.utils.data.DataLoader(
		valid,
		batch_size=batch_size_valid,
		shuffle=False,
	)

	#################################################################################################

	INPUT_SIZE = train[0][0].shape[0]
	logger.info(f"Number of training sources: {train.__len__()}")
	logger.info(f"Number of validation sources: {valid.__len__()}")
	logger.info(f"Size of one sample: {INPUT_SIZE}")

	test_params = {
		"test_name": TEST_NAME,
		"data_file": DATA,
		"flux_type": flux_type,
		"ae_type": args.model_type,
		"config": CONFIG,
		"latent_size": LATENT_SIZE,
		"activation_function": ACTIVATION_FUNCTION,
		"max_epochs": EPOCHS,
		"beta": BETA,
		"learn_rate": LEARNING_RATE,
		"weight_decay": WEIGHT_DECAY,
	}

	logging.info(test_params)

	funcs.save_test_params(test_params, TEST_NAME, test=TESTING)

	model = None
	if args.model_type == "StandardAutoencoder":
		model = ae.StandardAutoencoder(
			CONFIG, INPUT_SIZE, LATENT_SIZE, flux_type, normalize, activation=ACTIVATION_FUNCTION
		)
	elif args.model_type == "VariationalAutoencoder":
		model = ae.VAEAutoencoder(
			CONFIG, INPUT_SIZE, LATENT_SIZE, flux_type, normalize, activation=ACTIVATION_FUNCTION
		)
	else:
		exit()

	######################################################################################################
	### W AND B ###

	run = wandb.init(
		entity = "worrellie-iastro",
		project = args.project_name,
		# hyperparams and metadata
		config = test_params,
		name = TEST_NAME
	)
	wandb.config.update({
		"architecture" : CONFIG,
		"n_layers" : len(CONFIG),
		"normalize" : normalize,
		"input_size" : INPUT_SIZE,
		"early_stopping": EARLY_STOPPING,
		"n_train" : len(train),
		"n_valid" : len(valid),
		})


	######################################################################################################

	logger.info(model)
	wandb.config.update({"model_size_mb" : funcs.get_model_size_mb(model)})

	if EARLY_STOPPING:
		early_stopping = training.CustomEarlyStopping(
			TEST_NAME, patience=10, delta=0.0, test=TESTING, verbose=verb
		)
	else:
		early_stopping = None

	optimizer = optim.Adam(
		model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
	)

	# train
	torch.cuda.empty_cache()
	# model, model_losses = funcs.train_ae(EPOCHS, train_loader, valid_loader, model, optimizer, early_stopping = early_stopping, beta=BETA, verbose = verb, )
	start = time.time()
	trainer = training.Trainer(
		device, TEST_NAME, model, optimizer, early_stopping, BETA, test=TESTING
	)
	model, model_losses = trainer.train_ae(
		EPOCHS, train_loader, valid_loader=valid_loader, verbose=verb,
	)
	stop = time.time()

	wandb.log({"train_time" : stop - start})

	logger.info(f"{stop - start} seconds to train")

	funcs.plot_loss(model_losses, test_params["test_name"], test=TESTING)
	funcs.plot_dists(model_losses, test_params["test_name"], test=TESTING)

	train_fig_scaled, train_fig_unscaled, _ = funcs.plot_examples(train_loader, model, test_params, test=TESTING)
	wandb.log({"references/train_scaled":   wandb.Image(train_fig_scaled),
        	"references/train_unscaled": wandb.Image(train_fig_unscaled)})

	
	valid_fig_scaled, valid_fig_unscaled, valid_losses = funcs.plot_examples(valid_loader, model, test_params, test=TESTING)
	wandb.log({"references/valid_scaled":   wandb.Image(valid_fig_scaled),
        	"references/valid_unscaled": wandb.Image(valid_fig_unscaled),})

	wandb.log({
		# scaled space - for cross-run comparison
		"loss_dist/valid_scaled_mean":   valid_losses["scaled"]["mean"],
		"loss_dist/valid_scaled_median": valid_losses["scaled"]["median"],
		"loss_dist/valid_scaled_p95":    valid_losses["scaled"]["p95"],
		"loss_dist/valid_scaled_max":    valid_losses["scaled"]["max"],

		# unscaled space - physically meaningful
		"loss_dist/valid_unscaled_mean":   valid_losses["unscaled"]["mean"],
		"loss_dist/valid_unscaled_median": valid_losses["unscaled"]["median"],
		"loss_dist/valid_unscaled_p95":    valid_losses["unscaled"]["p95"],
		"loss_dist/valid_unscaled_max":    valid_losses["unscaled"]["max"],
	})
	# funcs.plot_examples(valid_loader, model, test_params, test = TESTING)


if __name__ == "__main__":
	main()
