import cv2
import numpy
import os
import collections
import operator
import math

# Used for timing
import time


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
    p1 = numpy.float32([kp.pt for kp in mkp1])
    p2 = numpy.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def match_image(image, matcher, kp_and_dest_pairs):
	# index:a sequence of descriptors
	# image:the image which is compared to match
	# kp_and_dest_pairs: the pairs of keypoints and descriptors
	# Get image descriptors
	image = get_image(image)
	keypoints, descriptors = get_image_features(image)

	h2, w2 = image.shape[:2]
	#corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])

	for kpd in kp_and_dest_pairs:
		imgSize = kpd[2]
		h1 = numpy.int(imgSize[0])
		w1 = numpy.int(imgSize[1])
		h3 = imgSize[0]
		corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])


		print kpd[3]," : ",len(kpd[0])
		raw_matches = matcher.knnMatch(kpd[1], descriptors, 2)
		p1, p2, kp_pairs = filter_matches(kpd[0], keypoints, raw_matches)

		#print '%s:' %kpd[2]

		if len(p1) >= 4:
			H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

			print '%s: %d / %d  inliers/matched,and the matched points is p1 = %d p2 = %d' % (kpd[2],numpy.sum(status), len(status),len(p1),len(p2))
		else:
			H, status = None, None
			print '%d matches found, not enough for homography estimation' % len(p1)

		if H is not None:
			corners = corners.reshape(1, -1, 2)
			print  "corners : ", corners
			#obj_corners = numpy.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))

			obj_corners = numpy.int32(cv2.perspectiveTransform(corners,H).reshape(-1,2))
			print "obj = " , obj_corners

			#corners = numpy .int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))

			vis = numpy.zeros((h2, w2), numpy.uint8)
			vis[:h2, :w2] = image
			vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

			cv2.polylines(vis, [obj_corners], True, (0, 0, 255))
			#print math.sqrt((obj_corners[0].x - obj_corners[1].x) * (obj_corners[0].x - obj_corners[1].x) + (
				#obj_corners[0].y - obj_corners[1].y) * (obj_corners[0].y - obj_corners[1].y))
			cv2.imshow(kpd[3], vis)
			cv2.waitKey()



	#vis = explore_match(win, kp_pairs, status, H)

	# # Find 2 closest matches for each descriptor in image
	# matches = index.knnMatch(descriptors, 2)
	#
	# # Cound matcher for each image in training set
	# print "Counting matches..."
	# count_dict = collections.defaultdict(int)
	# for match in matches:
	# 	# Only count as "match" if the two closest matches have big enough distance
	# 	if match[0].distance / match[1].distance < 0.7:
	# 		continue
    #
	# 	image_idx = match[0].imgIdx
	# 	count_dict[files[image_idx]] += 1

	# Get image with largest count
	# matched_image = max(count_dict.iteritems(), key=operator.itemgetter(1))[0]

	# Show results
	# print "Images", files
	# print "Counts: ", count_dict
	# print "==========="
	# print "Hit: ", matched_image
	# print "==========="
    #
	# return matched_image
def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = numpy.zeros((max(h1, h2), w1+w2), numpy.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = numpy.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = numpy.ones(len(kp_pairs), numpy.bool_)
    p1 = numpy.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = numpy.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

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

    cv2.imshow(win, vis)

if __name__ == "__main__":
	print "OpenCV Demo, OpenCV version " + cv2.__version__
	
	start_time = time.time()
	flann_matcher,kps,dests,sizes = train_index()  #get matcher with a sequence of desciptors #!!
	kp_and_dest_pairs = zip(kps,dests,sizes,files)
	#print kp_and_dest_pairs

	print "\nIndex generation took ", (time.time() - start_time), "s.\n"
	# ======================== Training done, image matching here ===============

	FLANN_INDEX_KDTREE = 1
	flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	matcher = cv2.FlannBasedMatcher(flann_params, {})

	surf = cv2.SURF()
	#surf = cv2.FeatureDetector_create("SURF")

	img1 = cv2.imread("tst.jpg", 0)
	img2 = cv2.imread("tst2.jpg", 0)
	kp1, desc1 = surf.detectAndCompute(img1, None)
	kp2, desc2 = surf.detectAndCompute(img2, None)
	raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)

	start_time = time.time()
	match_image("t3.jpg", flann_matcher, kp_and_dest_pairs)
	print "Matching took", (time.time() - start_time), "s."

