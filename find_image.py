import cv2
import numpy
import os
import collections
import operator

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

	# Train FLANN matcher with descriptors of all images
	for f in os.listdir("img/"):
		print "Processing " + f
		image = get_image("./img/%s" % (f,))
		#gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in image]
		#gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		#dst = cv2.fastNlMeansDenoisingMulti(image, 2, 5, None, 4, 7, 35)
		keypoints, descriptors = get_image_features(image)
		#!! matcher.add([descriptors])
		kps.add(keypoints) #!!
		dests.add(descriptors) #!!
		files.append(f)

	#!!print "Training FLANN."
	#!!matcher.train()
	#!!print "Done."
	return matcher,kps,dests

def match_image(image, matcher, kps, dests):
	# index:a sequence of descriptors
	# image:the image which is compared to match
	# Get image descriptors
	image = get_image(image)
	keypoints, descriptors = get_image_features(image)

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
	print "Images", files
	print "Counts: ", count_dict
	print "==========="
	print "Hit: ", matched_image
	print "==========="

	return matched_image

if __name__ == "__main__":
	print "OpenCV Demo, OpenCV version " + cv2.__version__
	
	start_time = time.time()
	flann_matcher,kps,dests = train_index()  #get matcher with a sequence of desciptors #!!
	kp_and_dest_pairs = zip(kps,dests)
	print "\nIndex generation took ", (time.time() - start_time), "s.\n"
	# ======================== Training done, image matching here ===============
	
	start_time = time.time()
	match_image("tst2.jpg", flann_matcher, kps, dests)
	print "Matching took", (time.time() - start_time), "s."

