import utils, debug
from config import *

import math
import cv2, numpy as np
import collections
na = np.array

NC_FAPL_CLAHE = [[2,   (2, 6),   15], # @1
		         [2,   (6, 2),   15], # @2
				 [0,   (0, 0),    0]] # EE

################################################################################

def fapl_canny(img, sigma=0.33):
	"""apply Canny edge detector (automatic thresh)"""
	v = np.median(img)
	img = cv2.medianBlur(img, 5)
	img = cv2.GaussianBlur(img, (7, 7), 2)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	return cv2.Canny(img, lower, upper)

def fapl_detector(img, alfa=150, beta=2):
	"""detect lines using Hough algorithm"""
	__lines, lines = [], cv2.HoughLinesP(img, rho=1, theta=np.pi/360*beta,
		threshold=40, minLineLength=50, maxLineGap=15) # [40, 40, 10]
	if lines is None: return []
	for line in np.reshape(lines, (-1, 4)):
		__lines += [[[int(line[0]), int(line[1])],
			         [int(line[2]), int(line[3])]]]
	return __lines

def fapl_clahe(img, limit=2, grid=(3,3), iters=5):
	"""repair using CLAHE algorithm (adaptive histogram equalization)"""
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	for i in range(iters):
		img = cv2.createCLAHE(clipLimit=limit, \
				tileGridSize=grid).apply(img)
	debug.image(img).save("fapl_clahe_@1")
	if limit != 0:
		kernel = np.ones((10, 10), np.uint8)
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		debug.image(img).save("fapl_clahe_@2")
	return img

################################################################################

def pFAPL(img, thresh=150):
	"""find all lines using different settings"""
	print(utils.call("pFAPL(img)"))
	segments = []; i = 0
	for key, arr in enumerate(NC_FAPL_CLAHE):
		tmp = fapl_clahe(img, limit=arr[0], grid=arr[1], iters=arr[2])
		__segments = list(fapl_detector(fapl_canny(tmp), thresh))
		segments += __segments; i += 1
		print("FILTER: {} {} : {}".format(i, arr, len(__segments)))
		debug.image(fapl_canny(tmp)).lines(__segments).save("pfapl_F%d" % i)
	return segments

def FAPL(img, segments):
	print(utils.call("FAPL(img, segments)"))

	pregroup, group, hashmap, raw_lines = [[], []], {}, {}, []

	__cache = {}
	def __dis(a, b):
		idx = hash("__dis" + str(a) + str(b))
		if idx in __cache: return __cache[idx]
		__cache[idx] = np.linalg.norm(na(a)-na(b))
		return __cache[idx]

	X = {}
	def __fi(x):
		if x not in X: X[x] = 0;
		if (X[x] == x or X[x] == 0): X[x] = x
		else:                        X[x] = __fi(X[x])
		return X[x]
	def __un(a, b):
		ia, ib = __fi(a), __fi(b)
		X[ia] = __fi(ib); group[ib] |= group[ia]

	# shortest path // height
	nln = lambda l1, x, dx: \
		np.linalg.norm(np.cross(na(l1[1])-na(l1[0]),
								na(l1[0])-na(   x)))/dx

	def __similar(l1, l2):
		da, db = __dis(l1[0], l1[1]), __dis(l2[0], l2[1])
		if da > db: l1, l2, da, db = l2, l1, db, da

		d1a, d2a = nln(l1, l2[0], da), nln(l1, l2[1], da)
		d1b, d2b = nln(l2, l1[0], db), nln(l2, l1[1], db)
		
		d1, d2 = (d1a + d1b)/2, (d2a + d2b)/2

		if d1 + d2 == 0: d1 += 0.00001 # [FIXME]: divide by 0
		t1 = (da/(d1 + d2) > 10 and db/(d1 + d2) > 10)
		if not t1: return False # [FIXME]: dist???
		return True

	def __generate(a, b, n):
		points = []; t = 1/n
		for i in range(n):
			x = a[0] + (b[0]-a[0]) * (i * t)
			y = a[1] + (b[1]-a[1]) * (i * t)
			points += [[int(x), int(y)]]
		return points

	def __analyze(group):
		points = []
		for idx in group:
			points += __generate(*hashmap[idx], 10)

		#debug.image(img.shape).points(points, \
		#	color=debug.color(), size=2).save("fapl__" + str(hash(str(group))))

		_, radius = cv2.minEnclosingCircle(na(points)); w = radius * (math.pi/2)
		vx, vy, cx, cy = cv2.fitLine(na(points), cv2.DIST_L2, 0, 0.01, 0.01)
		return [[int(cx-vx*w), int(cy-vy*w)], [int(cx+vx*w), int(cy+vy*w)]]

	for l in segments:
		h = hash(str(l))
		t1 = l[0][0] - l[1][0]
		t2 = l[0][1] - l[1][1]
		hashmap[h] = l; group[h] = set([h])
		if abs(t1) < abs(t2): pregroup[0].append(l)
		else:                 pregroup[1].append(l)

	debug.image(img.shape) \
		.lines(pregroup[0], color=debug.color()) \
		.lines(pregroup[1], color=debug.color()) \
	.save("fapl_pre_groups")

	for lines in pregroup:
		for i in range(len(lines)):
			for j in range(i+1, len(lines)):
				l1, l2 = lines[i], lines[j]
				if not __similar(l1, l2): continue
				h1, h2 = hash(str(l1)), hash(str(l2))
				__un(h1, h2) # union & find

	__d = debug.image(img.shape)
	for i in group:
		if (__fi(i) != i): continue
		ls = [hashmap[h] for h in group[i]]
		__d.lines(ls, color=debug.color())
	__d.save("fapl_all_groups")

	for i in group:
		if (__fi(i) != i): continue
		raw_lines += [__analyze(group[i])]
	debug.image(img.shape).lines(raw_lines).save("fapl_final")

	return raw_lines

def fapl_tendency(raw_lines, s=4): # FIXME: [1.25 -> 2]
	print(utils.call("fapl_tendency(raw_lines)"))
	lines = []; scale = lambda x, y, s: \
		int(x * (1+s)/2 + y * (1-s)/2)
	for a, b in raw_lines:
		# [A] s - scale
		# Xa' = Xa (1+s)/2 + Xb (1-s)/2
		# Ya' = Ya (1+s)/2 + Yb (1-s)/2
		a[0] = scale(a[0], b[0], s)
		a[1] = scale(a[1], b[1], s)
		# [B] s - scale
		# Xb' = Xb (1+s)/2 + Xa (1-s)/2
		# Yb' = Yb (1+s)/2 + Ya (1-s)/2
		b[0] = scale(b[0], a[0], s)
		b[1] = scale(b[1], a[1], s)
		lines += [[a, b]]
	return lines
