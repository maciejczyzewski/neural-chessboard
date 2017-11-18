from config import *
from random import randint
from copy import copy

import numpy as np
import cv2; load = cv2.imread
save = cv2.imwrite

################################################################################

def lines(img, lines, color=(0,0,255), size=2):
	"""draw lines"""
	for a, b in lines: cv2.line(img,tuple(a),tuple(b),color,size)
	return img

def points(img, points, color=(0,0,255), size=10):
	"""draw points"""
	for pt in points: cv2.circle(img,(int(pt[0]),int(pt[1])),size,color,-1)
	return img

def color():
	return (randint(0, 255), randint(0, 255), randint(0, 255))

################################################################################

counter = 0

class ImageDebug(object):
	img = object

	def __init__(self, img):
		if isinstance(img, tuple):
			img = np.zeros((img[0], img[1], 3), np.uint8)
		if len(img.shape) < 3:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		self.img = copy(img)

	def lines(self, _lines, color=(0,0,255), size=2):
		self.img = lines(self.img, _lines, color=color, size=size)
		return self

	def points(self, _points, color=(0,0,255), size=10):
		self.img = points(self.img, _points, color=color, size=size)
		return self

	def save(self, filename, prefix=True):
		global counter; counter += 1
		if prefix: __prefix = "__debug_"+"%04d"%int(counter)+"_"
		else:      __prefix = ""
		if NC_DEBUG: save("test/steps/" + __prefix + \
				filename + ".jpg", self.img)

image = ImageDebug
