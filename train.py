#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import sys, deps
import glob, os

import keras.models

print("---- FASTEN YOUR SEATBELTS -----") # FIXME
print("If it's slow, compile protobuf and tensorflow from source!")

NC_PATH_MODELS = 'data/models/'
NC_PATH_DATASET = 'data/train/'

NC_MODELS = {
	'LAPS': {'network': deps.laps.model, 'labels': None},
	'MAIN': {'network': None,            'labels': None}
}

################################################################################

def read_dataset(name):
	global NC_PATH_DATASET
	path = NC_PATH_DATASET + "{}.h5".format(name)
	h5f = h5py.File(path, 'r', driver='core')
	X, Y = h5f['data'], h5f['labels']
	X = X[()].reshape([-1, 21, 21, 1])
	Y = Y[()].reshape([-1, 2])
	return (X, Y)

def train_network(model, X, Y, n=50):
	if n == 0: return model
	model.fit(X, Y, epochs=n, batch_size=64, shuffle="batch")
	pred = model.predict(X); print("FINAL", np.mean(np.square(pred - Y)))
	return model

def load_model(name, best=False):
	global NC_MODELS, NC_PATH_MODELS
	model = NC_MODELS[name]['network']
	#best_path = NC_PATH_MODELS + '{}.weights.h5'.format(name.lower())
	#if best and os.path.isfile(best_path):
	#	model.load_weights(best_path)
	best_path = NC_PATH_MODELS + '{}.h5'.format(name.lower())
	if best and os.path.isfile(best_path):
		model = keras.models.load_model(best_path)
	return model

def save_model(name):
	global NC_PATH_MODELS
	save_path = NC_PATH_MODELS + '{}.h5'.format(name.lower())
	model.save(save_path)
	save_model = NC_PATH_MODELS + '{}.model.json'.format(name.lower())
	with open(save_model, "w") as json_file:
		json_file.write(model.to_json())
	save_weights = NC_PATH_MODELS + '{}.weights.h5'.format(name.lower())
	model.save_weights(save_weights)

################################################################################

print("[FIXME]: only LAPS model is supported")
NAME = 'LAPS'.upper()
model = train_network(load_model(NAME, best=True),
		*read_dataset(NAME), n=int(sys.argv[1]))
save_model(NAME)
