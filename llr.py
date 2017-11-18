import utils, debug

import scipy, cv2, pyclipper, numpy as np
import matplotlib.path, matplotlib.pyplot as plt
import matplotlib.path as mplPath
import collections, itertools, random, math
from copy import copy
na = np.array

from slid import slid_tendency
from laps import laps_intersections, laps_cluster

################################################################################

def llr_normalize(points): return [[int(a), int(b)] for a, b in points]

def llr_correctness(points, shape):
	__points = []
	for pt in points:
		if pt[0] < 0 or pt[1] < 0 or \
			pt[0] > shape[1] or \
			pt[1] > shape[0]: continue
		__points += [pt]
	return __points

def llr_unique(a):
	indices = sorted(range(len(a)), key=a.__getitem__)
	indices = set(next(it) for k, it in
		itertools.groupby(indices, key=a.__getitem__))
	return [x for i, x in enumerate(a) if i in indices]

def llr_polysort(pts):
	"""sort points clockwise"""
	mlat = sum(x[0] for x in pts) / len(pts)
	mlng = sum(x[1] for x in pts) / len(pts)
	def __sort(x): # main math --> found on MIT site
		return (math.atan2(x[0]-mlat, x[1]-mlng) + \
				2*math.pi)%(2*math.pi)
	pts.sort(key=__sort)
	return pts

def llr_polyscore(cnt, pts, alfa=5, beta=2):
	a = cnt[0]; b = cnt[1]
	c = cnt[2]; d = cnt[3]

	# (1) # za mala powierzchnia
	area = cv2.contourArea(cnt)
	t2 = area < (4 * alfa * alfa) * 5
	if t2: return 0

	# (2) # za malo punktow
	pco = pyclipper.PyclipperOffset()
	pco.AddPath(cnt, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
	pcnt = matplotlib.path.Path(pco.Execute(alfa/2)[0])
	pts_in = min(np.count_nonzero(pcnt.contains_points(pts)), 49)
	t1 = pts_in < min(len(pts), 49) - 1.5 * beta
	if t1: return 0
	
	# (3)
	# FIXME: punkty za kwadratowosci? (przypadki z L shape)

	A = pts_in * pts_in * pts_in
	B = area * area * math.log10(area) * \
		max(49 - pts_in, 1)

	if B == 0: return 0
	return A/B

################################################################################

# LAPS, SLID

def LLR(img, points, lines):
	print(utils.call("LLR(img, points, lines)"))

	# --- otoczka
	def __convex_approx(points, alfa=0.01):
		hull = scipy.spatial.ConvexHull(na(points)).vertices
		cnt = na([points[pt] for pt in hull])
		approx = cv2.approxPolyDP(cnt,alfa*\
				 cv2.arcLength(cnt,True),True)
		return llr_normalize(itertools.chain(*approx))
	# ---

	# --- geometria
	__cache = {}
	def __dis(a, b):
		idx = hash("__dis" + str(a) + str(b))
		if idx in __cache: return __cache[idx]
		__cache[idx] = np.linalg.norm(na(a)-na(b))
		return __cache[idx]

	nln = lambda l1, x, dx: \
		np.linalg.norm(np.cross(na(l1[1])-na(l1[0]),
								na(l1[0])-na(   x)))/dx
	# ---

	good_lines = []                       # nosnik "dobrych" linii
	pregroup = [[], []]                   # podzial na 2 grupy (dla ramki)
	S = {}                                # ranking ramek // wraz z wynikiem

	points = llr_correctness(llr_normalize(points), img.shape) # popraw punkty

	# --- clustrowanie
	import sklearn.cluster
	__points = {}; points = llr_polysort(points); __max, __points_max = 0, []
	alfa = math.sqrt(cv2.contourArea(na(points))/49)
	X = sklearn.cluster.DBSCAN(eps=alfa*3).fit(points) # **(1.3)
	for i in range(len(points)): __points[i] = []
	for i in range(len(points)):
		if X.labels_[i] != -1: __points[X.labels_[i]] += [points[i]]
	for i in range(len(points)):
		if len(__points[i]) > __max:
			__max = len(__points[i]); __points_max = __points[i]
	if len(__points) > 0 and len(points) > 49/2: points = __points_max
	print(X.labels_)
	# ---

	# tworzymy zewnetrzny pierscien
	ring = __convex_approx(llr_polysort(points))

	n = len(points); beta = n*(5/100) # beta=n*(100-(skutecznosc LAPS))
	alfa = math.sqrt(cv2.contourArea(na(points))/49) # srednia otoczka siatki

	x = [p[0] for p in points]          # szukamy punktu
	y = [p[1] for p in points]          # centralnego skupiska
	centroid = (sum(x) / len(points), \
			    sum(y) / len(points))

	print(alfa, beta, centroid)

	#        C (x2, y2)        d=(x_1−x_0)^2+(y_1−y_0)^2, t=d_t/d
	#      B (x1, y1)          (x_2,y_2)=(((1−t)x_0+tx_1),((1−t)y_0+ty_1))
	#    .                    t=(x_0-x_2)/(x_0-x_1)
	#  .
	# A (x0, y0)

	def __v(l):
		y_0, x_0 = l[0][0], l[0][1]
		y_1, x_1 = l[1][0], l[1][1]
		
		x_2 = 0;            t=(x_0-x_2)/(x_0-x_1+0.0001)
		a = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)][::-1]

		x_2 = img.shape[0]; t=(x_0-x_2)/(x_0-x_1+0.0001)
		b = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)][::-1]

		poly1 = llr_polysort([[0,0], [0, img.shape[0]], a, b])
		s1 = llr_polyscore(na(poly1), points, beta=beta, alfa=alfa/2)
		poly2 = llr_polysort([a, b, \
				[img.shape[1],0], [img.shape[1],img.shape[0]]])
		s2 = llr_polyscore(na(poly2), points, beta=beta, alfa=alfa/2)
		
		return [a, b], s1, s2

	def __h(l):
		x_0, y_0 = l[0][0], l[0][1]
		x_1, y_1 = l[1][0], l[1][1]
		
		x_2 = 0;            t=(x_0-x_2)/(x_0-x_1+0.0001)
		a = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)]

		x_2 = img.shape[1]; t=(x_0-x_2)/(x_0-x_1+0.0001)
		b = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)]

		poly1 = llr_polysort([[0,0], [img.shape[1], 0], a, b])
		s1 = llr_polyscore(na(poly1), points, beta=beta, alfa=alfa/2)
		poly2 = llr_polysort([a, b, \
				[0, img.shape[0]], [img.shape[1], img.shape[0]]])
		s2 = llr_polyscore(na(poly2), points, beta=beta, alfa=alfa/2)

		return [a, b], s1, s2

	for l in lines: # bedziemy wszystkie przegladac
		for p in points: # odrzucamy linie ktore nie pasuja
			# (1) linia przechodzi blisko dobrego punktu
			t1 = nln(l, p, __dis(*l)) < alfa
			# (2) linia przechodzi przez srodek skupiska
			t2 = nln(l, centroid, __dis(*l)) > alfa * 3 # 2.5
			# (3) linia nalezy do pierscienia
			t3 = True if p in ring else False
			if (t1 and t2) or (t1 and t3): # [1 and 2] or [1 and 3]
				tx, ty = l[0][0]-l[1][0], l[0][1]-l[1][1]
				if abs(tx) < abs(ty): ll, s1, s2 = __v(l); o = 0
				else:                 ll, s1, s2 = __h(l); o = 1
				if s1 == 0 and s2 == 0: continue
				pregroup[o] += [ll]

	pregroup[0] = llr_unique(pregroup[0])
	pregroup[1] = llr_unique(pregroup[1])

	print(alfa, beta)

	# (1) z jakiegos powodu mamy straszne szumy
	# if len(points) < 49/4 and len(lines) > 49/1.5:
	#	print("CAPTAIN, WE HAVE A PROBLEM!")
	#	rect = cv2.minAreaRect(na(llr_polysort(llr_normalize(ring))))
	#	box = cv2.boxPoints(rect); return llr_normalize(np.int0(box))

	debug.image(img) \
		.lines(lines, color=(0,0,255)) \
		.points(points, color=(0,0,255)) \
		.points(ring, color=(0,255,0)) \
		.points([centroid], color=(255,0,0)) \
	.save("llr_debug")
	
	debug.image(img) \
		.lines(pregroup[0], color=(0,0,255)) \
		.lines(pregroup[1], color=(255,0,0)) \
	.save("llr_pregroups")
	
	for v in itertools.combinations(pregroup[0], 2):            # poziome
		for h in itertools.combinations(pregroup[1], 2):        # pionowe
			poly = laps_intersections([v[0], v[1], h[0], h[1]]) # przeciecia
			poly = llr_correctness(poly, img.shape)             # w obrazku
			if len(poly) != 4: continue                         # jesl. nie ma
			poly = na(llr_polysort(llr_normalize(poly)))        # sortuj
			if not cv2.isContourConvex(poly): continue          # wypukly?
			S[-llr_polyscore(poly, points, \
				beta=beta, alfa=alfa/2)] = poly                 # dodaj

	S = collections.OrderedDict(sorted(S.items()))              # max
	four_points = llr_normalize(S[next(iter(S))])               # score

	print("POINTS:", len(points))
	print("LINES:", len(lines))
	print("GOOD:", len(good_lines))

	debug.image(img).points(four_points).save("llr_four_points")

	return four_points

def llr_pad(four_points):
	print(utils.call("llr_pad(four_points)"));pco = pyclipper.PyclipperOffset()
	pco.AddPath(four_points, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
	return pco.Execute(60)[0] # 70/75 is best (with buffer/for debug purpose)
