from config import *
from time import time
from copy import copy

import functools, os, re
import sys, cv2, math, numpy as np
na = np.array

################################################################################

rows, columns = os.popen('stty size', 'r').read().split()
__strip_ansi_re = re.compile(r"""
    \x1b     # literal ESC
    \[       # literal [
    [;\d]*   # zero or more digits or semicolons
    [A-Za-z] # a letter
    """, re.VERBOSE).sub
def __strip_ansi(s):
    return __strip_ansi_re("", s)

################################################################################

def clock():
	global NC_CLOCK; return "(%8s)s" % round((time() - NC_CLOCK), 3)
def reset(): global NC_CLOCK; NC_CLOCK = time()

def warn(msg): print("\x1b[0;33;40m warn: \x1b[4;33;40m" + msg + "\x1b[0m")
def errn(msg): print("\n\x1b[0;37;41m errn: " + msg + "\x1b[0m\n"); sys.exit(1)

def head(msg): return "\x1b[5;30;43m " + msg + " \x1b[0m"
def call(msg): return "--> \x1b[5;31;40m@" + msg + "\x1b[0m"

def ribb(*msg, sep='-'):
	msg = ' '.join(msg)
	return msg + sep * int(int(columns) - len(__strip_ansi(msg)))

################################################################################

def image_scale(pts, scale):
	"""scale to original image size"""
	def __loop(x, y): return [x[0] * y, x[1] * y]
	return list(map(functools.partial(__loop, y=1/scale), pts))

def image_resize(img, height=500):
	"""resize image to same normalized area (height**2)"""
	pixels = height * height; shape = list(np.shape(img))
	scale = math.sqrt(float(pixels)/float(shape[0]*shape[1]))
	shape[0] *= scale; shape[1] *= scale
	img = cv2.resize(img, (int(shape[1]), int(shape[0])))
	img_shape = np.shape(img)
	return img, img_shape, scale

def image_transform(img, points, square_length=150):
	"""crop original image using perspective warp"""
	board_length = square_length * 8
	def __dis(a, b): return np.linalg.norm(na(a)-na(b))
	def __shi(seq, n=0): return seq[-(n % len(seq)):] + seq[:-(n % len(seq))]
	best_idx, best_val = 0, 10**6
	for idx, val in enumerate(points):
		val = __dis(val, [0, 0])
		if val < best_val:
			best_idx, best_val = idx, val
	pts1 = np.float32(__shi(points, 4 - best_idx))
	pts2 = np.float32([[0, 0], [board_length, 0], \
			[board_length, board_length], [0, board_length]])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	W = cv2.warpPerspective(img, M, (board_length, board_length))
	return W

class ImageObject(object):
	images = {}; scale = 1; shape = (0, 0)

	def __init__(self, img):
		"""save and prepare image array"""
		self.images['orig'] = img
		self.images['main'], self.shape, self.scale = \
				image_resize(img) # downscale for speed
		self.images['test'] = copy(self.images['main'])

	def __getitem__(self, attr):
		"""return image as array"""
		return self.images[attr]

	def __setitem__(self, attr, val):
		"""save image to object"""
		self.images[attr] = val

	def crop(self, pts):
		"""crop using 4 points transform"""
		pts_orig = image_scale(pts, self.scale)
		img_crop = image_transform(self.images['orig'], pts_orig)
		self.__init__(img_crop)
