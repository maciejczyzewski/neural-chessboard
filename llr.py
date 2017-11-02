import utils, debug

import scipy, cv2, pyclipper, numpy as np
import matplotlib.path, matplotlib.pyplot as plt
import matplotlib.path as mplPath
import collections, itertools, random, math
from copy import copy
na = np.array

from fapl import fapl_tendency
from pamg import pamg_intersections, pamg_cluster

################################################################################

def llr_normalize(points): return [[int(a), int(b)] for a, b in points]
def llr_exodus(a, b): return [pt for pt in a if pt not in b]

def llr_net(grid, points, alfa=1.1):
	fail = []

	def __get(cnt):
		rect = cv2.minAreaRect(na(cnt))
		box = cv2.boxPoints(rect)
		return np.int0(box)

	for pt in points:
		net = grid + [pt]
		cnt1, cnt2 = __get(grid), __get(net)

		area1 = cv2.contourArea(cnt1)
		area2 = cv2.contourArea(cnt2)

		n, a = len(grid), abs(area2 - area1)
		g1, g2 = area1/n, area2/(n + 1)

		if g1 * alfa > g2: grid = net
		else: fail += [pt]

	return (grid, fail)

def llr_prediction(points):
	lines = []; n = len(points)
	for k1 in range(0, n):
		for k2 in range(k1 + 1, k1 + 4): # [FIXME]: uwazac na to
			lines += [[list(points[k1 % n]), \
					   list(points[k2 % n])]]
	return lines

def llr_correctness(points, shape):
	__points = []
	for pt in points:
		if pt[0] < 0 or pt[1] < 0 or \
			pt[0] > shape[1] or \
			pt[1] > shape[0]: continue
		__points += [pt]
	return __points

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

	pco = pyclipper.PyclipperOffset()
	pco.AddPath(cnt, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
	pcnt = matplotlib.path.Path(pco.Execute(alfa)[0])
	pts_in = np.count_nonzero(pcnt.contains_points(pts))
	
	if len(pts) - beta > pts_in: return 0

	# [FIXME]: jak eliminowac wiele zlych punktow??? wagi???
	area = cv2.contourArea(cnt)

	#x, y, w, h = cv2.boundingRect(cnt)
	#box = na([[x,y], [x+w, y], [x+w, y+h], [x, y+h]])
	#area = cv2.contourArea(box)

	A = pts_in * pts_in * pts_in
	B = area * math.log10(area)
	
	return A/B

################################################################################

def LLR(img, points):
	print(utils.call("LLR(img, points)"))

	points = llr_normalize(points)
	all_points, grid = copy(points), copy(points)

	debug.image(img).points(all_points).save("llr_all_points")

	def __convex(points):
		hull = scipy.spatial.ConvexHull(na(points)).vertices
		return llr_normalize([points[pt] for pt in hull])

	def __convex_approx(points, alfa=0.01):
		hull = scipy.spatial.ConvexHull(na(points)).vertices
		cnt = na([points[pt] for pt in hull])
		approx = cv2.approxPolyDP(cnt,alfa*\
				 cv2.arcLength(cnt,True),True)
		return llr_normalize(itertools.chain(*approx))

	cnt1 = __convex(grid)
	grid = llr_exodus(grid, cnt1)

	cnt2 = __convex(grid)
	grid = llr_exodus(grid, cnt2)
	
	ring = cnt1 + cnt2

	debug.image(img).points(grid).save("llr_inner_grid")

	grid, ring = llr_net(grid, ring, alfa=1.1)
	grid, ring = llr_net(grid, ring, alfa=1.3)

	debug.image(img).points(grid).save("llr_extended_grid")
	
	cnt1 = __convex(grid)
	cnt2 = __convex(llr_exodus(grid, cnt1))
	
	ring = cnt1 + cnt2
	
	# --- DANGER ZONE ---
	n = len(grid); beta = n*(5/100)
	alfa = cv2.contourArea(na(cnt1))/n/n/3 # 3
	# --- DANGER ZONE ---

	approx_ring = __convex_approx(ring, 0.001)

	debug.image(img).points(approx_ring).save("llr_approx_ring")

	lines    = llr_prediction(approx_ring)
	phantoms = pamg_intersections(fapl_tendency(lines, s=20))

	debug.image(img).points(phantoms).save("llr_phantoms_@1")

	# --- DANGER ZONE ---
	pco = pyclipper.PyclipperOffset()
	pco.AddPath(na(approx_ring), pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
	cnt_approx = pco.Execute(alfa)[0]
	bbPath, __phantoms = mplPath.Path(na(cnt_approx)), []
	for pt in phantoms:
		if not bbPath.contains_point((pt[0], pt[1])): __phantoms += [pt]
	phantoms = __phantoms
	# --- DANGER ZONE ---

	if len(phantoms) > 2:
		phantoms = llr_correctness(pamg_cluster(phantoms), img.shape)

	debug.image(img).points(phantoms).save("llr_phantoms_@2")
	
	outer_ring = llr_normalize(pamg_cluster(phantoms + ring)) # [FIXME]

	debug.image(img).points(outer_ring).save("llr_outer_ring")

	S = {}
	for poly in itertools.combinations(outer_ring, 4):
		poly = na(llr_polysort(llr_normalize(poly)))
		if not cv2.isContourConvex(poly): continue
		S[-llr_polyscore(poly, grid, \
			beta=beta, alfa=alfa/2)] = poly

	S = collections.OrderedDict(sorted(S.items()))
	four_points = llr_normalize(S[next(iter(S))])

	print("ALFA", alfa, "BETA", beta)

	debug.image(img).points(four_points).save("llr_four_points")

	return four_points

def llr_pad(four_points):
	print(utils.call("llr_pad(four_points)"));pco = pyclipper.PyclipperOffset()
	pco.AddPath(four_points, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
	return pco.Execute(60)[0] # 70/75 is best (with buffer/for debug purpose)
