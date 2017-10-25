import cv2
import timeit
from numpy import math, hstack

import numpy as np


class FileVideoCapture(object):

	def __init__(self, path):
		self.path = path
		self.frame = 1

	def isOpened(self):
		im = cv2.imread(self.path.format(self.frame))
		return im != None

	def read(self):
		im = cv2.imread(self.path.format(self.frame))
		status = im != None
		if status:
			self.frame += 1
		return status, im

def squeeze_pts(X):
	X = X.squeeze()
	if len(X.shape) == 1:
		X = np.array([X])
	return X

def array_to_int_tuple(X):
	return (int(X[0]), int(X[1]))

def L2norm(X):
	return np.sqrt((X ** 2).sum(axis=1))

current_pos = None
tl = None
br = None

def get_rect(im, title='get_rect'):
	mouse_params = {'tl': None, 'br': None, 'current_pos': None,
		'released_once': False}

	cv2.namedWindow(title)
	cv2.moveWindow(title, 100, 100)

	def onMouse(event, x, y, flags, param):

		param['current_pos'] = (x, y)

		if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
			param['released_once'] = True

		if flags & cv2.EVENT_FLAG_LBUTTON:
			if param['tl'] is None:
				param['tl'] = param['current_pos']
			elif param['released_once']:
				param['br'] = param['current_pos']

	cv2.setMouseCallback(title, onMouse, mouse_params)
	cv2.imshow(title, im)

	while mouse_params['br'] is None:
		im_draw = np.copy(im)

		if mouse_params['tl'] is not None:
			cv2.rectangle(im_draw, mouse_params['tl'],
				mouse_params['current_pos'], (255, 0, 0))

		cv2.imshow(title, im_draw)
		_ = cv2.waitKey(10)

	cv2.destroyWindow(title)

	tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
		min(mouse_params['tl'][1], mouse_params['br'][1]))
	br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
		max(mouse_params['tl'][1], mouse_params['br'][1]))

	return (tl, br)

def time_execution(callable):
	start_time = timeit.default_timer()
	results = callable()
	elapsed = timeit.default_timer - start_time
	return results, elapsed

def in_rect(keypoints, tl, br):
	if type(keypoints) is list:
		keypoints = keypoints_cv_to_np(keypoints)

	x = keypoints[:, 0]
	y = keypoints[:, 1]

	C1 = x > tl[0]
	C2 = y > tl[1]
	C3 = x < br[0]
	C4 = y < br[1]

	result = C1 & C2 & C3 & C4

	return result

def keypoints_cv_to_np(keypoints_cv):
	keypoints = np.array([k.pt for k in keypoints_cv])
	return keypoints

def find_nearest_keypoints(keypoints, pos, number=1):
	if type(pos) is tuple:
		pos = np.array(pos)
	if type(keypoints) is list:
		keypoints = keypoints_cv_to_np(keypoints)

	pos_to_keypoints = np.sqrt(np.power(keypoints - pos, 2).sum(axis=1))
	ind = np.argsort(pos_to_keypoints)
	return ind[:number]

def draw_keypoints(keypoints, im, color=(255, 0, 0)):

	for k in keypoints:
		radius = 3  # int(k.size / 2)
		center = (int(k[0]), int(k[1]))

		# Draw circle
		cv2.circle(im, center, radius, color)

def track(im_prev, im_gray, keypoints, THR_FB=20):
	if type(keypoints) is list:
		keypoints = keypoints_cv_to_np(keypoints)

	num_keypoints = keypoints.shape[0]

	# Status of tracked keypoint - True means successfully tracked
	status = [False] * num_keypoints

	# If at least one keypoint is active
	if num_keypoints > 0:
		# Prepare data for opencv:
		# Add singleton dimension
		# Use only first and second column
		# Make sure dtype is float32
		pts = keypoints[:, None, :2].astype(np.float32)

		# Calculate forward optical flow for prev_location
		nextPts, status, _ = cv2.calcOpticalFlowPyrLK(im_prev, im_gray, pts, None)

		# Calculate backward optical flow for prev_location
		pts_back, _, _ = cv2.calcOpticalFlowPyrLK(im_gray, im_prev, nextPts, None)

		# Remove singleton dimension
		pts_back = squeeze_pts(pts_back)
		pts = squeeze_pts(pts)
		nextPts = squeeze_pts(nextPts)
		status = status.squeeze()

		# Calculate forward-backward error
		fb_err = np.sqrt(np.power(pts_back - pts, 2).sum(axis=1))

		# Set status depending on fb_err and lk error
		large_fb = fb_err > THR_FB
		status = ~large_fb & status.astype(np.bool)

		nextPts = nextPts[status, :]
		keypoints_tracked = keypoints[status, :]
		keypoints_tracked[:, :2] = nextPts

	else:
		keypoints_tracked = np.array([])
	return keypoints_tracked, status

def rotate(pt, rad):
	if(rad == 0):
		return pt

	pt_rot = np.empty(pt.shape)

	s, c = [f(rad) for f in (math.sin, math.cos)]

	pt_rot[:, 0] = c * pt[:, 0] - s * pt[:, 1]
	pt_rot[:, 1] = s * pt[:, 0] + c * pt[:, 1]

	return pt_rot

def br(bbs):

	result = hstack((bbs[:, [0]] + bbs[:, [2]] - 1, bbs[:, [1]] + bbs[:, [3]] - 1))

	return result

def bb2pts(bbs):

	pts = hstack((bbs[:, :2], br(bbs)))

	return pts

def to_tl_br(bbox):
	"""Convert (x, y, w, h) bbox to (x1, y1, x2, y2), where (x1, y1) = top left corner, (x2, y2) = bottom right.

    Args:
        bbox: (int, int, int, int): Bounding box around target in (x, y, w, h)
            format. x and y are image coordinates of the top left corner of the bounding box.
            w and h are the width and height of the bounding box respectively.
    """

	left, top, width, height = bbox
	return (left, top), (left + width, top + height)

def to_xywh(tl, br):
	"""Convert (x1, y1, x2, y2) bbox to (x, y, w, h).

    Args:
        tl (int, int): The coordinates of the top-left corner of the bounding box.
        br (int, int): The coordinates of the bottom-right corner of the bounding box.
    """

	left, top = tl
	right, bottom = br
	return left, top, right - left, bottom - top

def region_to_bbox(region, center=True):

    n = len(region)
    assert n==4 or n==8, ('GT region format is invalid, should have 4 or 8 entries.')

    if n==4:
        return _rect(region, center)
    else:
        return _poly(region, center)

# we assume the grountruth bounding boxes are saved with 0-indexing
def _rect(region, center):

    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
        return cx, cy, w, h
    else:
        #region[0] -= 1
        #region[1] -= 1
        return region


def _poly(region, center):
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1/A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return cx, cy, w, h
    else:
        return cx-w/2, cy-h/2, w, h

def compile_results(ground_truth_regions, predicted_bboxes, dist_threshold):
	l = np.size(predicted_bboxes, 0)
	ground_truth_bboxes = np.zeros((l, 4))
	new_distances = np.zeros(l)
	new_ious = np.zeros(l)
	n_thresholds = 50
	precisions_ths = np.zeros(n_thresholds)

	# Compute IoUs for each frame.
	for i in range(l):
		ground_truth_bboxes[i, :] = region_to_bbox(ground_truth_regions[i, :], center=False)
		new_distances[i] = _compute_distance(predicted_bboxes[i, :], ground_truth_bboxes[i, :])
		new_ious[i] = _compute_iou(predicted_bboxes[i, :], ground_truth_bboxes[i, :])

	# what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
	precision = sum(new_distances < dist_threshold)/np.size(new_distances) * 100

	# find above result for many thresholds, then report the AUC
	thresholds = np.linspace(0, 25, n_thresholds+1)
	thresholds = thresholds[-n_thresholds:]
	# reverse it so that higher values of precision goes at the beginning
	thresholds = thresholds[::-1]
	for i in range(n_thresholds):
		precisions_ths[i] = sum(new_distances < thresholds[i])/np.size(new_distances)

	# integrate over the thresholds
	precision_auc = np.trapz(precisions_ths)

	# per frame averaged intersection over union (OTB metric)
	iou = np.mean(new_ious) * 100

	return l, precision, precision_auc, iou

def _compute_distance(boxA, boxB):
	a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
	b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
	dist = np.linalg.norm(a - b)

	assert dist >= 0
	assert dist != float('Inf')

	return dist


def _compute_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
	yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

	if xA < xB and yA < yB:
		# compute the area of intersection rectangle
		interArea = (xB - xA) * (yB - yA)
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = boxA[2] * boxA[3]
		boxBArea = boxB[2] * boxB[3]
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the intersection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
	else:
		iou = 0

	assert iou >= 0
	assert iou <= 1.01

	return iou
