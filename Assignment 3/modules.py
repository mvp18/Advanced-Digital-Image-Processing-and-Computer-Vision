import cv2
import matplotlib.pyplot as plt

def FLANN_matcher(img1, img2, kp1, kp2, des1, des2):

	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=100)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1, des2, k=2)

	# Need to draw only good matches, so create a mask
	matchesMask = [[0,0] for i in range(len(matches))]
	num_matches=0
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.7*n.distance:
			print('M:')
			print(m.trainIdx, m.queryIdx, m.imgIdx)
			print('N:')
			print(n.trainIdx, n.queryIdx, n.imgIdx)
			matchesMask[i]=[1,0]
			num_matches+=1

	print(num_matches)
	draw_params = dict(matchColor = (0,255,0),
					   singlePointColor = (0,0,255),
					   matchesMask = matchesMask,
					   flags = 0)

	img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

	return img3

def brute_force_matcher(img1, img2, kp1, kp2, des1, des2):

	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	# Match descriptors.
	matches = bf.match(des1, des2)
	print("Number of matching keypoints found:{}".format(len(matches)))
	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	
	draw_params = dict(matchColor = (0,255,0), singlePointColor = (0,0,255), flags = 0)

	X1 = []
	X2 = []
	
	# Paramters of Keypoint class - pt, size (diameter of the meaningful keypoint neighborhood), angle
	# Paramters of DMatch object:
	# DMatch.distance - Distance between descriptors. The lower, the better it is.
	# DMatch.trainIdx - Index of the descriptor in train descriptors
	# DMatch.queryIdx - Index of the descriptor in query descriptors
	# DMatch.imgIdx - Index of the train image.
	
	for i, m in enumerate(matches[:8]):
		# print(match.trainIdx, match.queryIdx)
		X1.append((kp1[m.queryIdx].pt[1], kp1[m.queryIdx].pt[0]))
		X2.append((kp2[m.trainIdx].pt[1], kp2[m.trainIdx].pt[0]))

	img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:8], None, **draw_params)
	
	return img3, X1, X2
	# plt.imshow(img3)
	# plt.show()

def compute_SIFT_ckp(img1, img2, save_flag):
	
	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	sift = cv2.xfeatures2d.SIFT_create()

	kp1, des1 = sift.detectAndCompute(gray1, None)
	kp2, des2 = sift.detectAndCompute(gray2, None)

	img1 = cv2.drawKeypoints(gray1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	img2 = cv2.drawKeypoints(gray2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	cv2.imshow('SIFT_keypoints1', img1)
	cv2.imshow('SIFT_keypoints2', img2)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	if save_flag:
		cv2.imwrite('sift_keypoints1.png', img1)
		cv2.imwrite('sift_keypoints2.png', img2)

	img3, X1, X2 = brute_force_matcher(gray1, gray2, kp1, kp2, des1, des2)
	
	# cv2.imshow('matched_pts_sift', img3)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	if save_flag:
		cv2.imwrite('matched_pts_sift.png', img3)

	return X1, X2

def compute_SURF_ckp(img1, img2, save_flag):

	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	surf = cv2.xfeatures2d.SURF_create(400, True)

	kp1, des1 = surf.detectAndCompute(gray1, None)
	kp2, des2 = surf.detectAndCompute(gray2, None)

	img1 = cv2.drawKeypoints(gray1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	img2 = cv2.drawKeypoints(gray2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	cv2.imshow('SURF_keypoints1', img1)
	cv2.imshow('SURF_keypoints2', img2)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	if save_flag:
		cv2.imwrite('surf_keypoints1.png', img1)
		cv2.imwrite('surf_keypoints2.png', img2)

	img3, X1, X2 = brute_force_matcher(gray1, gray2, kp1, kp2, des1, des2)

	cv2.imshow('matched_pts_surf', img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	if save_flag:
		cv2.imwrite('matched_pts_surf.png', img3)

	return X1, X2