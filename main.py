import os, sys, glob, argparse, utils
print("<<< \x1b[5;32;40m neural-chessboard \x1b[0m >>>")

from config import *
from utils import ImageObject
from fapl import pFAPL, FAPL, fapl_tendency #== step 1
from pamg import PAMG                       #== step 2
from llr import LLR, llr_pad                #== step 3

import cv2; load = cv2.imread
save = cv2.imwrite

################################################################################

def layer():
	global NC_LAYER, NC_IMAGE
	
	print(utils.ribb("==", sep="="))
	print(utils.ribb("[%d] LAYER " % NC_LAYER, sep="="))
	print(utils.ribb("==", sep="="), "\n")

	# --- 1 step --- find all possible lines (that makes sense) ----------------
	print(utils.ribb(utils.head("FAPL"), utils.clock(), "--- 1 step "))
	segments = pFAPL(NC_IMAGE['main'])
	raw_lines = FAPL(NC_IMAGE['main'], segments)
	lines = fapl_tendency(raw_lines)

	# --- 2 step --- find interesting intersections (potentially a mesh grid) --
	print(utils.ribb(utils.head("PAMG"), utils.clock(), "--- 2 step "))
	points = PAMG(NC_IMAGE['main'], lines)

	# --- 3 step --- last layer reproduction (for chessboard corners) ----------
	print(utils.ribb(utils.head(" LLR"), utils.clock(), "--- 3 step "))
	four_points = llr_pad(LLR(NC_IMAGE['main'], points))

	# --- 4 step --- preparation for next layer (deep analysis) ----------------
	print(utils.ribb(utils.head("   *"), utils.clock(), "--- 4 step "))
	print(four_points)
	try: NC_IMAGE.crop(four_points)
	except: utils.warn("niestety, ale kolejna warstwa nie jest potrzebna")

	print("\n")

################################################################################

def detect(args):
	global NC_LAYER, NC_IMAGE, NC_CONFIG

	if (not os.path.isfile(args.input)):
		utils.errn("error: the file \"%s\" does not exits" % args.input)

	NC_IMAGE, NC_LAYER = ImageObject(load(args.input)), 0
	for _ in range(NC_CONFIG['layers']):
		NC_LAYER += 1; layer()
	save(args.output, NC_IMAGE['orig'])

	print("DETECT: %s" % args.input)

def dataset(args):
	print("DATASET: use dataset.py") # FIXME

def train(args):
	print("TRAIN: use train.py") # FIXME

def test(args):
	files = glob.glob('test/in/*.jpg')

	for iname in files:
		oname = iname.replace('in', 'out')
		args.input = iname; args.output = oname
		detect(args)

	print("TEST: %d images" % len(files))
	
################################################################################

if __name__ == "__main__":
	utils.reset()

	p = argparse.ArgumentParser(description=\
	'Find, crop and create FEN from image.')

	p.add_argument('mode', nargs=1, type=str, \
			help='detect | dataset | train')
	p.add_argument('--input', type=str, \
			help='input image (default: input.jpg)')
	p.add_argument('--output', type=str, \
			help='output path (default: output.jpg)')

	os.system("rm test/steps/*.jpg") # FIXME: to jest bardzo grozne

	args = p.parse_args(); mode = str(args.mode[0])
	modes = {'detect': detect, 'dataset': dataset, 'train': train, 'test': test}

	if mode not in modes.keys():
		utils.errn("hey, nie mamy takiej procedury!!! (wybrano: %s)" % mode)

	modes[mode](args); print(utils.clock(), "done")
