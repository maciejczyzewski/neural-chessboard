import utils, debug, deps

import collections
import cv2, numpy as np
import scipy, scipy.cluster
from config import *

from keras.models import model_from_json
__laps_model = 'data/models/laps.model.json'
__laps_weights = 'data/models/laps.weights.h5'
NC_LAPS_MODEL = model_from_json(open(__laps_model, 'r').read())
NC_LAPS_MODEL.load_weights(__laps_weights)

#from keras.utils import plot_model, print_summary
#plot_model(NC_LAPS_MODEL, show_shapes=True, to_file='model.png')
#print_summary(NC_LAPS_MODEL)

################################################################################

def laps_intersections(lines):
	"""find all intersections"""
	__lines = [[(a[0], a[1]), (b[0], b[1])] for a, b in lines]
	return deps.geometry.isect_segments(__lines)

def laps_cluster(points, max_dist=10):
	"""cluster very similar points"""
	Y = scipy.spatial.distance.pdist(points)
	Z = scipy.cluster.hierarchy.single(Y)
	T = scipy.cluster.hierarchy.fcluster(Z, max_dist, 'distance')
	clusters = collections.defaultdict(list)
	for i in range(len(T)):
		clusters[T[i]].append(points[i])
	clusters = clusters.values()
	clusters = map(lambda arr: (np.mean(np.array(arr)[:,0]),
		                        np.mean(np.array(arr)[:,1])), clusters)
	return list(clusters) # if two points are close, they become one mean point

def laps_detector(img):
	"""determine if that shape is positive"""
	global NC_LAYER

	hashid = str(hash(img.tostring()))

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
	img = cv2.Canny(img, 0, 255)	
	img = cv2.resize(img, (21, 21), interpolation=cv2.INTER_CUBIC)
	
	imgd = img

	X = [np.where(img>int(255/2), 1, 0).ravel()]
	X = X[0].reshape([-1, 21, 21, 1])

	img = cv2.dilate(img, None)
	mask = cv2.copyMakeBorder(img, top=1, bottom=1, left=1, right=1,
		borderType=cv2.BORDER_CONSTANT, value=[255,255,255])
	mask = cv2.bitwise_not(mask); i = 0
	_1, contours, _2 = cv2.findContours(mask,cv2.RETR_EXTERNAL,
				                             cv2.CHAIN_APPROX_NONE)
	
	_c = np.zeros((23,23,3), np.uint8)

	# geometric detector
	for cnt in contours:
		(x,y),radius = cv2.minEnclosingCircle(cnt); x,y=int(x),int(y)
		approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
		if len(approx) == 4 and radius < 14:
			cv2.drawContours(_c, [cnt], 0, (0,255,0), 1)
			i += 1
		else:
			cv2.drawContours(_c, [cnt], 0, (0,0,255), 1)
	
	if i == 4: return (True, 1)

	pred = NC_LAPS_MODEL.predict(X)
	a, b = pred[0][0], pred[0][1]
	t = a > b and b < 0.03 and a > 0.975

	# decision
	if t:
		#debug.image(imgd).save("OK" + str(hash(str(imgd))), prefix=False)
		return (True, pred[0])
	else:
		#debug.image(imgd).save("NO" + str(hash(str(imgd))), prefix=False)
		return (False, pred[0])

################################################################################

def LAPS(img, lines, size=10):
	print(utils.call("LAPS(img, lines)"))

	__points, points = laps_intersections(lines), []
	debug.image(img).points(__points, size=3).save("laps_in_queue")

	for pt in __points:
		# pixels are in integers
		pt = list(map(int, pt))

		# size of our analysis area
		lx1 = max(0, int(pt[0]-size-1)); lx2 = max(0, int(pt[0]+size))
		ly1 = max(0, int(pt[1]-size)); ly2 = max(0, int(pt[1]+size+1))

		# cropping for detector
		dimg = img[ly1:ly2, lx1:lx2]
		dimg_shape = np.shape(dimg)
		
		# not valid
		if dimg_shape[0] <= 0 or dimg_shape[1] <= 0: continue

		# use neural network
		re_laps = laps_detector(dimg)
		if not re_laps[0]: continue

		# add if okay
		if pt[0] < 0 or pt[1] < 0: continue
		points += [pt]
	points = laps_cluster(points)

	debug.image(img).points(points, size=5, \
		color=debug.color()).save("laps_good_points")
	
	return points
