import utils, debug
from config import *

import math
import cv2, numpy as np
import collections
na = np.array

"""
NC_SLID_CLAHE = [[3,   (2, 6),    5], # @1
		         [3,   (6, 2),    5], # @2
				 [0,   (0, 0),    0]] # EE
"""

"""
NC_SLID_CLAHE = [[4,   (2, 5),    5], # @1
		         [4,   (5, 2),    5], # @2
				 #[1,   (2, 2),   15], # @3
				 [0,   (0, 0),    0]] # EE
"""

"""
NC_SLID_CLAHE = [[2,   (1, 5),    5], # @1
		         [2,   (5, 1),    5], # @2
				 #[1,   (2, 2),   15], # @3
				 [0,   (0, 0),    0]] # EE
"""

"""
NC_SLID_CLAHE = [[3,   (2, 8),    5], # @1
		         [3,   (8, 2),    5], # @2
				 [5,   (4, 4),    5], # @3
				 [0,   (0, 0),    0]] # EE
"""

# 7???
# 4???
NC_SLID_CLAHE = [[3,   (2, 6),    5], # @1
		         [3,   (6, 2),    5], # @2
				 [5,   (3, 3),    5], # @3
				 [0,   (0, 0),    0]] # EE

################################################################################

def slid_canny(img, sigma=0.25):
	"""apply Canny edge detector (automatic thresh)"""
	v = np.median(img)
	img = cv2.medianBlur(img, 5)
	img = cv2.GaussianBlur(img, (7, 7), 2)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	return cv2.Canny(img, lower, upper)

def slid_detector(img, alfa=150, beta=2):
	"""detect lines using Hough algorithm"""
	__lines, lines = [], cv2.HoughLinesP(img, rho=1, theta=np.pi/360*beta,
		threshold=40, minLineLength=50, maxLineGap=15) # [40, 40, 10]
	if lines is None: return []
	for line in np.reshape(lines, (-1, 4)):
		__lines += [[[int(line[0]), int(line[1])],
			         [int(line[2]), int(line[3])]]]
	return __lines

def slid_clahe(img, limit=2, grid=(3,3), iters=5):
	"""repair using CLAHE algorithm (adaptive histogram equalization)"""
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	for i in range(iters):
		img = cv2.createCLAHE(clipLimit=limit, \
				tileGridSize=grid).apply(img)
	debug.image(img).save("slid_clahe_@1")
	if limit != 0:
		kernel = np.ones((10, 10), np.uint8)
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		debug.image(img).save("slid_clahe_@2")
	return img

################################################################################

def pSLID(img, thresh=150):
	"""find all lines using different settings"""
	print(utils.call("pSLID(img)"))
	segments = []; i = 0
	for key, arr in enumerate(NC_SLID_CLAHE):
		tmp = slid_clahe(img, limit=arr[0], grid=arr[1], iters=arr[2])
		__segments = list(slid_detector(slid_canny(tmp), thresh))
		segments += __segments; i += 1
		print("FILTER: {} {} : {}".format(i, arr, len(__segments)))
		debug.image(slid_canny(tmp)).lines(__segments).save("pslid_F%d" % i)
	return segments

all_points = []
def SLID(img, segments):
	# FIXME: zrobic 2 rodzaje haszowania (katy + pasy [blad - delta])
	print(utils.call("SLID(img, segments)"))
	
	global all_points; all_points = []
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
		X[ia] = ib; group[ib] |= group[ia]
		#group[ia] = set()
		#group[ia] = set()

	# shortest path // height
	nln = lambda l1, x, dx: \
		np.linalg.norm(np.cross(na(l1[1])-na(l1[0]),
								na(l1[0])-na(   x)))/dx

	def __similar(l1, l2):
		da, db = __dis(l1[0], l1[1]), __dis(l2[0], l2[1])
		# if da > db: l1, l2, da, db = l2, l1, db, da

		d1a, d2a = nln(l1, l2[0], da), nln(l1, l2[1], da)
		d1b, d2b = nln(l2, l1[0], db), nln(l2, l1[1], db)
	
		ds = 0.25 * (d1a + d1b + d2a + d2b) + 0.00001
		#print(da, db, abs(da-db))
		#print(int(da/ds), int(db/ds), "|", int(abs(da-db)), int(da+db),
		#		int(da+db)/(int(abs(da-db))+0.00001))
		alfa = 0.0625 * (da + db) #15
		# FIXME: roznica???
		#if d1 + d2 == 0: d1 += 0.00001 # [FIXME]: divide by 0
		t1 = (da/ds > alfa and db/ds > alfa)
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
		global all_points
		points = []
		for idx in group:
			points += __generate(*hashmap[idx], 10)
		_, radius = cv2.minEnclosingCircle(na(points)); w = radius * (math.pi/2)
		vx, vy, cx, cy = cv2.fitLine(na(points), cv2.DIST_L2, 0, 0.01, 0.01)
		# debug.color()
		all_points += points
		return [[int(cx-vx*w), int(cy-vy*w)], [int(cx+vx*w), int(cy+vy*w)]]

	for l in segments:
		h = hash(str(l))
		t1 = l[0][0] - l[1][0]
		t2 = l[0][1] - l[1][1]
		hashmap[h] = l; group[h] = set([h]); X[h] = h
		if abs(t1) < abs(t2): pregroup[0].append(l)
		else:                 pregroup[1].append(l)

	debug.image(img.shape) \
		.lines(pregroup[0], color=debug.color()) \
		.lines(pregroup[1], color=debug.color()) \
	.save("slid_pre_groups")

	for lines in pregroup:
		for i in range(len(lines)):
			l1 = lines[i]; h1 = hash(str(l1))
			#print(h1, __fi(h1))
			if (X[h1] != h1): continue
			#if (__fi(h1) != h1): continue
			for j in range(i+1, len(lines)):
				l2 = lines[j]; h2 = hash(str(l2))
				#if (__fi(h2) != h2): continue
				if (X[h2] != h2): continue
				#if (len(group[h2])==0): continue
				if not __similar(l1, l2): continue
				__un(h1, h2) # union & find
				# break # FIXME

	__d = debug.image(img.shape)
	for i in group:
		#if (__fi(i) != i): continue
		if (X[i] != i): continue
		#if len(group[i]) == 0: continue
		ls = [hashmap[h] for h in group[i]]
		__d.lines(ls, color=debug.color())
	__d.save("slid_all_groups")

	for i in group:
		#if (__fi(i) != i): continue
		if (X[i] != i): continue
		#if len(group[i]) == 0: continue
		#if (__fi(i) != i): continue
		raw_lines += [__analyze(group[i])]
	debug.image(img.shape).lines(raw_lines).save("slid_final")

	debug.image(img.shape)\
		.points(all_points, color=(0,255,0), size=2)\
	.lines(raw_lines).save("slid_final2")

	return raw_lines

def slid_tendency(raw_lines, s=4): # FIXME: [1.25 -> 2]
	print(utils.call("slid_tendency(raw_lines)"))
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
