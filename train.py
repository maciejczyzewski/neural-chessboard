#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py, tflearn
import numpy as np
import sys, deps
import glob, os

print("---- FASTEN YOUR SEATBELTS -----") # FIXME
print("If it's slow, compile protobuf and tensorflow from source!")

# FIXME: cli + tensorboard (background process)
# FIXME: dataset preparation from "d" script (labels)
# FIXME: it should be more general (MAIN model compt.)

NC_PATH_MODELS = 'data/models/'
NC_PATH_DATASET = 'data/train/'

NC_MODELS = {
	'PAMG': {'network': deps.pamg.network(), 'labels': None},
	'MAIN': {'network': None,                'labels': None}
}

################################################################################

def read_dataset(name):
	global NC_PATH_DATASET
	path = NC_PATH_DATASET + "{}.hdf5".format(name)
	h5f = h5py.File(path, 'r', driver='core')
	X, Y = h5f['data'], h5f['labels']
	X = X[()].reshape([-1, 21, 21, 1])
	return (X, Y)

def train_network(model, X, Y, n=50):
	model.fit(X, Y, show_metric=True, snapshot_step=20, \
					n_epoch=n, validation_set=0.4, shuffle=True)
	# validation_set=0.2, shuffle=True (BEST)
	return model

def load_model(name, best=False):
	global NC_MODELS, NC_PATH_MODELS
	model = tflearn.DNN(NC_MODELS[name]['network'], tensorboard_verbose=3,
		best_checkpoint_path=NC_PATH_MODELS + '{}-'.format(name.lower()))
	best_path = NC_PATH_MODELS + '{}.tflearn'.format(name.lower())
	if best and os.path.isfile(best_path): model.load(best_path)
	return model

def save_model(name):
	global NC_PATH_MODELS
	prefix = NC_PATH_MODELS + '{}-'.format(name.lower())
	paths = glob.glob(prefix + '*.index')
	if len(paths) == 0: return False
	paths = [p.replace(prefix, '')\
			  .replace('.index', '') for p in paths]
	paths = sorted(map(int, paths))[::-1]; model_id = paths[0]
	print('BEST: \t{}%'.format(float(model_id/100)))
	files = ['data-00000-of-00001', 'index', 'meta']
	for ext in files:
		src = prefix + '{}.{}'.format(model_id, ext)
		dst = NC_PATH_MODELS + '{}.tflearn.{}'.format(name.lower(), ext)
		os.system('cp {} {}'.format(src, dst))
		print('cp', src, dst)
	paths.remove(model_id)
	for mid in paths:
		for ext in files:
			tmp = prefix + '{}.{}'.format(mid, ext)
			os.system('rm {}'.format(tmp))
			print('rm', tmp)
	return True

################################################################################

print("[FIXME]: only PAMG model is supported")
NAME = 'PAMG'.upper()
model = train_network(load_model(NAME, best=True),
		*read_dataset(NAME), n=int(sys.argv[1]))
save_model(NAME)
