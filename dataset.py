#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import shuffle
import h5py, cv2, glob
import numpy as np

shuffle_data = True # option

dataset_path  = 'data/train/laps.h5'
ok_train_path = 'data/train/laps/ok/*.jpg'
no_train_path = 'data/train/laps/no/*.jpg'

ok_addrs = glob.glob(ok_train_path) # 0
no_addrs = glob.glob(no_train_path) # 1

addrs = ok_addrs + no_addrs
labels = [[0, 1] if 'no' in addr else [1, 0] for addr in addrs]

if shuffle_data:
	c = list(zip(addrs, labels)); shuffle(c)
	addrs, labels = zip(*c)

train_addrs  = addrs[0:int(1 * len(addrs))]
train_labels = labels[0:int(1 * len(labels))]
train_shape  = (len(train_addrs), 21 * 21)

hdf5_file = h5py.File(dataset_path, mode='w')
hdf5_file.create_dataset("data", train_shape, np.int8)
hdf5_file.create_dataset("labels", (len(train_addrs), 2), np.int8)
hdf5_file["labels"][...] = train_labels

for i in range(len(train_addrs)):
	if i % 10 == 0 and i > 1:
		print('train data: {}/{}'.format(i, len(train_addrs)))
	addr = train_addrs[i]
	
	img = cv2.imread(addr) # PREPROCESSING
	img = cv2.resize(img, (21, 21), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = np.where(img>int(255/2), 1, 0).ravel()

	hdf5_file["data"][i, ...] = img

hdf5_file.close()

# check dataset correctness
"""
import glob
import deps as dd

def i(a, b): list(set(a) & set(b))

a = glob.glob("data/train/ok/*.jpg")
b = glob.glob("data/train/no/*.jpg")

print(len(a), len(b))
print(i(a, b))
"""
