import cv2
import numpy
import os

# Used for timing
import time
from common import anorm

#TRAINDIR = 'img'
TRAINDIR = 'test'
files = []
matcher = None

def get_image(image_path):
	return cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

def get_image_features(image):
	# Workadound for missing interfaces
	surf = cv2.FeatureDetector_create("SURF")
	surf.setInt("hessianThreshold", 1000)
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
	images = []

	# Train FLANN matcher with descriptors of all images
	for f in os.listdir(TRAINDIR):
		# This step may produce a .DS_Store file in OS X system,Please remove the file in ./img folder.
		print "Processing " + f

		if f == ".DS_Store":
			continue
		image = get_image(TRAINDIR + "/%s" % (f,))
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
		images.append(image)

	#!!print "Training FLANN."
	#!!matcher.train()
	#!!print "Done."
	return matcher,kps,dests,sizes,images

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    # kp1: keypoints of the first image
    # kp2: keypoints of the second image
    # matches: raw matches.
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx].pt )
            mkp2.append( kp2[m.trainIdx].pt )
    p1 = numpy.float32(mkp1)
    p2 = numpy.float32(mkp2)

    return p1, p2

def m_and_d_filter_matches(kp1, kp2, matches, ratio = 0.75):
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
	# image:the image which is compared to match
	# kp_and_dest_pairs: the pairs of keypoints, descriptors, iamge sizes and fileName
	# Get image descriptors
	queryImage = get_image(image)
	keypoints, descriptors = get_image_features(queryImage)

	h2, w2 = queryImage.shape[:2]

	for kpd in kp_and_dest_pairs:
		imgSize = kpd[2]
		h1 = numpy.int(imgSize[0])
		w1 = numpy.int(imgSize[1])
		trainImageName = kpd[3]
		#
		corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])

		print trainImageName," : ",len(kpd[0])
		raw_matches = matcher.knnMatch(kpd[1], descriptors, 2)
		p1, p2 = filter_matches(kpd[0], keypoints, raw_matches)

		#print '%s:' %kpd[2]

		if len(p1) >= 15:
			H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

			print "H: ", H
			print '%s: %d / %d  inliers/matched,and the matched points is p1 = %d p2 = %d' % (kpd[2], numpy.sum(status), len(status),len(p1),len(p2))
		else:
			H, status = None, None
			print '%d matches found, not enough for homography estimation' % len(p1)

		if H is not None:
			corners = corners.reshape(1, -1, 2)
			print  "corners : ", corners

			obj_corners = numpy.int32(cv2.perspectiveTransform(corners,H).reshape(-1,2))
			print "obj = " , obj_corners

			#corners = numpy .int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))


			# ================= just show the query image =================
			vis = numpy.zeros((h2, w2), numpy.uint8)
			vis[:h2, :w2] = queryImage
			vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
			cv2.polylines(vis, [obj_corners], True, (0, 0, 255))
			# =================== test for cv2.polylines ===================
            # p1 = [100, 100]
            # p2 = [100, 200]
            # p3 = [200, 200]
            # p4 = [200, 100]
            #
            # points = numpy.int32([p1, p2, p3, p4])
            #
            # red = (0 , 0 , 255)
            # green = (0, 255, 0)
            # blue = (255, 0, 0)
            # sky_blue = (255, 255, 0)
            #
            # color = [red, green, blue, sky_blue]
            #
            # #col = red
            # r = 2
            # thickness = 3
            #
            # for (x1,y1), col in zip(points,color):
				# cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
				# cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            #
            # cv2.polylines(vis,[points],True,(0,255,255))
			is_polygon = ispolygon(obj_corners)

			if is_polygon:
				print "This image may be matched!"
			else:
				print "This image may NOT be matched!"

			cv2.imshow(trainImageName, vis)
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

def match_and_draw(queryImageName, matcher, kp_dest_and_images_pairs):
	print 'match_and_draw...'
	queryImage = get_image(queryImageName)
	queryKeypoints, queryDescriptors = get_image_features(queryImage)

	for kpd in kp_dest_and_images_pairs:
		trainKeypoints = kpd[0]
		trainDescriptors = kpd[1]
		trainImgSize = kpd[2]
		h1 = numpy.int(trainImgSize[0])
		w1 = numpy.int(trainImgSize[1])
		trainImageName = kpd[3]
		trainImage = kpd[4]

		corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])

		print trainImageName," : ",len(trainKeypoints)
		raw_matches = matcher.knnMatch(trainDescriptors, queryDescriptors, 2)
		queryGoodKeypoints, trainGoodKeypoints, kp_pairs = m_and_d_filter_matches(trainKeypoints, queryKeypoints, raw_matches)

		if len(trainGoodKeypoints) >= 15:
			H, status = cv2.findHomography(queryGoodKeypoints, trainGoodKeypoints, cv2.RANSAC, 5.0)

			#print "H: ", H
			print '%s: %d / %d  inliers/matched,and the matched points is p1 = %d p2 = %d' % \
				  (kpd[2],numpy.sum(status), len(status),len(queryGoodKeypoints),len(trainGoodKeypoints))
		else:
			H, status = None, None
			print '%d matches found, not enough for homography estimation' % len(queryGoodKeypoints)

		if H is not None:
			corners = corners.reshape(1, -1, 2)
			print  "corners : ", corners

			obj_corners = numpy.int32(cv2.perspectiveTransform(corners,H).reshape(-1,2))
			print "obj = " , obj_corners

			is_polygon = ispolygon(obj_corners)

			if is_polygon:
				print "This image may be matched!"
			else:
				print "This image may NOT be matched!"

		explore_match(trainImageName, queryImage, trainImage, kp_pairs, status, H)


def explore_match(visName, queryImage, trainImage, kp_pairs, status = None, H = None):
	h1, w1 = queryImage.shape[:2]
	h2, w2 = trainImage.shape[:2]
	vis = numpy.zeros((max(h1, h2), w1+w2), numpy.uint8)
	vis[:h1, :w1] = queryImage
	vis[:h2, w1:w1+w2] = trainImage
	vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

	if H is not None:
		corners = numpy.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
		corners = numpy.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))
		cv2.polylines(vis, [corners], True, (0, 0, 255))

	if status is None:
		status = numpy.ones(len(kp_pairs), numpy.bool_)
	trainGKP = numpy.int32([kpp[1].pt for kpp in kp_pairs])		    #train good matched keypoints
	queryGKP = numpy.int32([kpp[0].pt for kpp in kp_pairs]) + (w1, 0) #query good matched keypoints

	green = (0, 255, 0)
	red = (0, 0, 255)
	white = (255, 255, 255)
	kp_color = (51, 103, 236)
	for (x1, y1), (x2, y2), inlier in zip(trainGKP, queryGKP, status):
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
	for (x1, y1), (x2, y2), inlier in zip(trainGKP, queryGKP, status):
		if inlier:
			cv2.line(vis, (x1, y1), (x2, y2), green)

	cv2.imshow(visName, vis)

	def onmouse(event, x, y, flags, param):
		cur_vis = vis
		if flags & cv2.EVENT_FLAG_LBUTTON:
			cur_vis = vis0.copy()
			r = 8
			m = (anorm(trainGKP - (x, y)) < r) | (anorm(queryGKP - (x, y)) < r)
			idxs = numpy.where(m)[0]
			kp1s, kp2s = [], []
			for i in idxs:
				(x1, y1), (x2, y2) = trainGKP[i], queryGKP[i]
				col = (red, green)[status[i]]
				cv2.line(cur_vis, (x1, y1), (x2, y2), col)
				kp1, kp2 = kp_pairs[i]
				kp1s.append(kp1)
				kp2s.append(kp2)
			cur_vis = cv2.drawKeypoints(cur_vis, kp2s, flags=4, color=kp_color)
			cur_vis[:, w1:] = cv2.drawKeypoints(cur_vis[:, w1:], kp1s, flags=4, color=kp_color)
		cv2.imshow(visName, cur_vis)
	cv2.setMouseCallback(visName, onmouse)

	cv2.waitKey(0)

	return vis

def match(queryFeature, trainFeature, matcher, queryImage = None):
	queryKeypoints = queryFeature[0]
	queryDescriptors = queryFeature[1]
	queryImgSize = queryFeature[2]
	queryImageName = queryFeature[3]
	queryImgHeight = queryImgSize[0]
	queryImgWidth = queryImgSize[1]

	trainKeypoints = trainFeature[0]
	trainDescriptors = trainFeature[1]
	trainImgSize = trainFeature[2]
	trainImageName = trainFeature[3]
	trainImgHeight = trainImgSize[0]
	trainImgWidth = trainImgSize[1]

	red = (0, 0, 255)

	corners = numpy.float32([[0,0],[trainImgWidth,0],[trainImgWidth,trainImgHeight],[0,trainImgHeight]])

	raw_matches = matcher.knnMatch(trainDescriptors, queryDescriptors, 2)
	queryGoodPoints , trainGoodPoints = filter_matches(trainKeypoints, queryKeypoints, raw_matches)

	if len(queryKeypoints) >= 4:
		H, status = cv2.findHomography(queryGoodPoints, trainGoodPoints, cv2.RANSAC, 5.0)

		print "H: ", H
		print '%s: %d / %d  inliers/matched,and the matched points is p1 = %d p2 = %d' % (
			trainImageName, numpy.sum(status), len(status), len(queryGoodPoints), len(trainGoodPoints))
	else:
		H, status = None, None
		print '%d matches found, not enough for homography estimation' % len(trainKeypoints)

	if H is not None:
		corners = corners.reshape(1, -1, 2)
		print  "corners : ", corners

		obj_corners = numpy.int32(cv2.perspectiveTransform(corners, H).reshape(-1, 2))

		print "obj = ", obj_corners


		# =========== just for test ============
		if queryImage != None:
			vis = numpy.zeros((queryImgHeight, queryImgWidth), numpy.uint8)
			vis[:queryImgHeight, :queryImgWidth] = queryImage
			vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
			cv2.polylines(vis, [obj_corners], True, red)
			is_polygon = ispolygon(obj_corners)

			if is_polygon:
				print "This image may be matched!"
			else:
				print "This image may NOT be matched!"
		# print math.sqrt((obj_corners[0].x - obj_corners[1].x) * (obj_corners[0].x - obj_corners[1].x) + (
		# obj_corners[0].y - obj_corners[1].y) * (obj_corners[0].y - obj_corners[1].y))
			cv2.imshow(trainImageName, vis)
			cv2.waitKey()

def absCosVector(x,y):
	l=len(x)
	# len(x)
	if(len(x)!=len(y)):
		print 'error input,x and y is not in the same space'
		return -9

	result1 = 0.0
	result2 = 0.0
	result3 = 0.0

	for i in range(l):
		result1+=x[i]*y[i]	#sum(x * y)
		result2+=x[i]**2	#sum(x * x)
		result3+=y[i]**2	#sum(y * y)

	if result2 == 0 or result3 == 0:
		return -10
	else:
		# print "result1 = ",result1
		# print "result2 = ",result2
		# print "result3 = ",result3
		# print("result is "+str(cosVec))
		absCosVec = abs(result1 / ((result2 * result3) ** 0.5))
		#print "absCosVec = ", absCosVec
		return absCosVec

def vector(p1,p2):

	if len(p1) == len(p2):
		v = []
		for i in range(len(p1)):
			v.append(p1[i] - p2[i])
		return v
	else:
		print "Error:len(p1) != len(p2)"
		return

def ispolygon(points):
	vec1 = vector(points[0], points[1])
	vec2 = vector(points[0], points[3])
	vec3 = vector(points[1], points[2])
	vec4 = vector(points[2], points[3])

	absCos = []
	absCos.append(absCosVector(vec1, vec2))
	absCos.append(absCosVector(vec1, vec3))
	absCos.append(absCosVector(vec1, vec4))
	absCos.append(absCosVector(vec2, vec4))
	absCos.append(absCosVector(vec2, vec3))
	absCos.append(absCosVector(vec3, vec4))
	approxVertical, approxHorizontal = 0, 0
	for cosine in absCos:
		if cosine < 0.26:
			approxVertical = approxVertical + 1
		if cosine >= 0.95 and cosine <= 1:
			approxHorizontal = approxHorizontal + 1

	print "absCos = ", absCos
	print "approxVertical[2,4]: ", approxVertical
	print "approxHorizontal[1,2]: ", approxHorizontal
	if approxVertical >= 2  and approxVertical <= 4 and approxHorizontal >= 1 and approxHorizontal <= 2:
		print "This area is polygon-like."
		return True
	else:
		return False

if __name__ == "__main__":
	print "OpenCV Demo, OpenCV version " + cv2.__version__

	# ======= test for ispolygon() =======
	# p1 = [1,1]
	# p2 = [200,1]
	# p3 = [200,200]
	# p4 = [1,200]
    #
	# points = [p1,p2,p3,p4]
	# ispolygon(points)

	start_time = time.time()
	flann_matcher,kps,dests,sizes,images = train_index()  #get matcher with a sequence of desciptors #!!

	kp_dest_and_images_pairs = zip(kps,dests,sizes,files,images)

	#print kp_and_dest_pairs

	print "\nIndex generation took ", (time.time() - start_time), "s.\n"
	# ======================== Training done, image matching here ===============

	FLANN_INDEX_KDTREE = 1
	flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
	matcher = cv2.FlannBasedMatcher(flann_params, {})

	# surf = cv2.SURF()
	# #surf = cv2.FeatureDetector_create("SURF")
    #
	# img1 = cv2.imread("tst.jpg", 0)
	# img2 = cv2.imread("tst2.jpg", 0)
	# kp1, desc1 = surf.detectAndCompute(img1, None)
	# kp2, desc2 = surf.detectAndCompute(img2, None)
	# raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)

	# ================== first match test ===================

	queryImage = get_image("t1.jpg")
	queryKeypoints, queryDescriptors = get_image_features(queryImage)
	queryImgSize = queryImage.shape[:2]
	queryFeature = [queryKeypoints, queryDescriptors, queryImgSize, "t1.jpg"]

	trainImage = get_image("1.jpg")
	trainKeypoints, trainDescriptors = get_image_features(trainImage)
	trainImgSize = trainImage.shape[:2]
	trainFeature = [trainKeypoints, trainDescriptors , trainImgSize, "1.jpg"]

	surf_extractor = cv2.DescriptorExtractor_create("SURF")

  	#bowDE = cv2.BOWImgDescriptorExtractor(surf_extractor, matcher)

	match(queryFeature,trainFeature, matcher, queryImage)
	#======================================================
	start_time = time.time()

	#================== second match test =================
	#match_image("t1.jpg", flann_matcher, kp_dest_and_images_pairs)
	print "Matching took", (time.time() - start_time), "s."

	#================== third match test ==================
	match_and_draw("IMG1.jpg", matcher, kp_dest_and_images_pairs)