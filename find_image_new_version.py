#!/usr/bin/env python

'''
Feature-based image matching sample.

USAGE
  find_obj.py [--feature=<sift|surf|orb>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf of orb. Append '-flann' to feature name
                to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

import numpy as np
import cv2
import os
import time
#from common import anorm, getsize

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6
files = []
matcher = None

def get_image(image_path):
	return cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

def get_image_features(image):
	# Workadound for missing interfaces
	surf = cv2.FeatureDetector_create("SURF")
	surf.setInt("hessianThreshold", 100)
	surf_extractor = cv2.DescriptorExtractor_create("SURF")
	# Get keypoints from image
	keypoints = surf.detect(image, None)
	# Get keypoint descriptors for found keypoints
	keypoints, descriptors = surf_extractor.compute(image, keypoints)
	return keypoints, numpy.array(descriptors)

# create a matcher with a sequence of all descriptors of images.
def train_index():
	# Prepare FLANN matcher
	#FLANN_INDEX_KDTREE = 0
	#flann_params = dict(algorithm = 1, trees = 4)
	flann_params = dict(algorithm = 1, trees = 5)
	#search_params = dict(checks = 50)
	matcher = cv2.FlannBasedMatcher(flann_params, {})
	kps = []
	dests = []
	sizes = []

	# Train FLANN matcher with descriptors of all images
	for f in os.listdir("img/"):
		# This step may produce a .DS_Store file in OS X system,Please remove the file in ./img folder.
		print "Processing " + f

		if f == ".DS_Store":
			continue
		image = get_image("./img/%s" % (f,))
		imgSize = image.shape[:2]

		#gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in image]
		#gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		#dst = cv2.fastNlMeansDenoisingMulti(image, 2, 5, None, 4, 7, 35)
		keypoints, descriptors = get_image_features(image)
		#!! matcher.add([descriptors])
		kps.append(keypoints) #!!
		dests.append(descriptors) #!!
		files.append(f)
		sizes.append(imgSize)

	#!!print "Training FLANN."
	#!!matcher.train()
	#!!print "Done."
	return matcher,kps,dests,sizes

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    # kp1: keypoints of the first image
    # kp2: keypoints of the second image
    # matches: raw matches.
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.SIFT()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.SURF(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB(400)
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio = 0.75):
    # kp1: keypoints of the first image
    # kp2: keypoints of the second image
    # matches: raw matches.
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def explore_match(size, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        print "corners : " , corners
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    #cv2.imshow(win, vis)
    # def onmouse(event, x, y, flags, param):
    #     cur_vis = vis
    #     if flags & cv2.EVENT_FLAG_LBUTTON:
    #         cur_vis = vis0.copy()
    #         r = 8
    #         m = (anorm(p1 - (x, y)) < r) | (anorm(p2 - (x, y)) < r)
    #         idxs = np.where(m)[0]
    #         kp1s, kp2s = [], []
    #         for i in idxs:
    #              (x1, y1), (x2, y2) = p1[i], p2[i]
    #              col = (red, green)[status[i]]
    #              cv2.line(cur_vis, (x1, y1), (x2, y2), col)
    #              kp1, kp2 = kp_pairs[i]
    #              kp1s.append(kp1)
    #              kp2s.append(kp2)
    #         cur_vis = cv2.drawKeypoints(cur_vis, kp1s, flags=4, color=kp_color)
    #         cur_vis[:,w1:] = cv2.drawKeypoints(cur_vis[:,w1:], kp2s, flags=4, color=kp_color)
    #
    #     cv2.imshow(win, cur_vis)
    # cv2.setMouseCallback(win, onmouse)
    return vis


if __name__ == '__main__':

    print "OpenCV Demo, OpenCV version " + cv2.__version__
    print __doc__

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    try: fn1, fn2 = args
    except:
        #fn1 = '../c/box.png'
        #fn2 = '../c/box_in_scene.png'
        fn1 = 'ad007.jpg'
        fn2 = 't007.jpg' #test

    img1 = cv2.imread(fn1, 0)
    img2 = cv2.imread(fn2, 0)
    size = img2.shape[:2]
    print "size : " ,size
    detector, matcher = init_feature(feature_name) #create detector and matcher object
    if detector != None:
        print 'using', feature_name
    else:
        print 'unknown feature:', feature_name
        sys.exit(1)


    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

    def match_and_draw(win):
        print 'matching...'
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        else:
            H, status = None, None
            print '%d matches found, not enough for homography estimation' % len(p1)

        vis = explore_match(win, img1, img2, kp_pairs, status, H)

    match_and_draw('find_obj')

    # ===========================================================================

    start_time = time.time()
    flann_matcher, kps, dests, sizes = train_index()  # get matcher with a sequence of desciptors #!!
    kp_and_dest_pairs = zip(kps, dests, sizes, files)
    # print kp_and_dest_pairs

    print "\nIndex generation took ", (time.time() - start_time), "s.\n"
    # ======================== Training done, image matching here ===============

    FLANN_INDEX_KDTREE = 1
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})

    start_time = time.time()
    match_image("t3.jpg", flann_matcher, kp_and_dest_pairs)
    print "Matching took", (time.time() - start_time), "s."
    cv2.waitKey()