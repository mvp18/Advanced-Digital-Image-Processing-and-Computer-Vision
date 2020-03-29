import numpy as np
import cv2

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

def draw_epipolar_lines(X1, X2, F, img1, img2, save_flag):

	L = []
	L_ = []
	img3 = img1
	img4 = img2
	
	for i in range(len(X1)):
		x = np.array([X1[i][0], X1[i][1], 1]).reshape(3, 1)
		x_ = np.array([X2[i][0], X2[i][1], 1]).reshape(3, 1)
		l = np.matmul(F.transpose(), x_)
		l_ = np.matmul(F, x)
		L.append(l)
		L_.append(l_)
		img3 = cv2.line(img3, (0, int(-l[2]/l[0])), (int(-l[2]/l[1]), 0), (0, 0, 255), 1)
		img4 = cv2.line(img4, (0, int(-l_[2]/l_[0])), (int(-l_[2]/l_[1]), 0), (0, 255, 0), 1)

	AB = np.array([[L[0][0], L[0][1]],
				   [L[1][0], L[1][1]]])
	
	C = np.array([L[0][2], L[1][2]])
	e = -np.matmul(np.linalg.inv(AB.reshape(2, 2)), C)
	e = np.append(e, 1)
	print('\nLeft Epipole from Epipolar Lines:{}'.format(e))
	
	AB_ = np.array([[L_[0][0], L_[0][1]],
				   [L_[1][0], L_[1][1]]])
	C_ = np.array([L_[0][2], L_[1][2]])
	e_ = -np.matmul(np.linalg.inv(AB_.reshape(2, 2)), C_)
	e_ = np.append(e_, 1)
	print('Right Epipole from Epipolar Lines1:{}'.format(e_))

	cv2.imshow('Epipolar_lines1', img3)
	cv2.imshow('Epipolar_lines2', img4)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	if save_flag:
		cv2.imwrite('Epipolar_lines1.png', img3)
		cv2.imwrite('Epipolar_lines2.png', img4)

	return e, e_

def give_scene_points(X1, X2, P1, P2):
	
    X = []
    
    for i in range(len(X1)):
        x1 = np.array([X1[i][0], X1[i][1]])
        x2 = np.array([X2[i][0], X2[i][1]])

        A = np.array([
            x1[0]*P1[2] - P1[0],
            x1[1]*P1[2] - P1[1],
            x2[0]*P2[2] - P2[0],
            x2[1]*P2[2] - P2[1]
        ])

        U, D, VT = np.linalg.svd(A)
        V = VT.transpose()

        X_3d = V[:,-1] / V[-1,-1]
        X_3d = X_3d[:-1]

        X.append(X_3d)
    
    return X