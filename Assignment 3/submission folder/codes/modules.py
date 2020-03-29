from utils import *

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
	
	cv2.imshow('matched_pts_sift', img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

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

def solve_8pt_corr(X1, X2):

	A = np.array([[X1[i][0]*X2[i][0], X2[i][0]*X1[i][1], X2[i][0], X1[i][0]*X2[i][1], 
				   X1[i][1]*X2[i][1], X2[i][1], X1[i][0], X1[i][1]] for i in range(len(X1))])
	f = np.ones([8, 1])*(-1)
	F = np.matmul(np.linalg.inv(A), f)
	F = np.append(F, 1)
	F = F.reshape(3, 3)
	U, D, VT = np.linalg.svd(F)
	D_ = np.array([[D[0], 0, 0],
				   [0, D[1], 0],
				   [0, 0, 0]])

	F_ = np.matmul(U, np.matmul(D_, VT))
	print('Determinant of fundamental matrix : {}'.format(np.linalg.det(F_)))

	return F_

def calc_epipoles_from_F(F):

	e = -np.matmul(np.linalg.inv(F[:2, :2]), F[:2, 2])
	e = np.append(e, 1)
	print('\nLeft Epipole from F:{}'.format(e))
	
	F_T = F.transpose()
	e_ = -np.matmul(np.linalg.inv(F_T[:2, :2]), F_T[:2, 2])
	e_ = np.append(e_, 1)
	print('Right Epipole from F:{}'.format(e_))

	return e, e_

def estimate_proj_matrices(F, e_):

	P = np.zeros((3, 4))

	P[:, :-1] = np.identity(3)

	S = np.array([[0, -e_[2], e_[1]],
		          [e_[2], 0, -e_[0]],
		          [-e_[1], e_[0], 0]])

	P_ = np.zeros((3, 4))
	P_[:, :-1] = np.matmul(S, F)
	P_[:, -1] = e_

	print('\nCamera Matrix P:\n{}'.format(P))
	print('\nCamera Matrix P`:\n{}\n'.format(P_))

	return P, P_

def estimate_scene_depth(img1, img2, X1, X2, P1, P2, save_flag):

	img3 = img1
	img4 = img2

	X = give_scene_points(X1, X2, P1, P2)

	colors = [(0,0,255), (0,255,255), (0,255,0), (255,255,0), (255,0,0), (255,0,255), (255,204,204), (255,255,255)]
	names = ['Red', 'Yellow', 'Green', 'Indigo', 'Blue', 'Pink', 'Light Pink', 'White']

	for i in range(len(X1)):
		x1, y1 = X1[i]
		x2, y2 = X2[i]

		img3 = cv2.circle(img3, (int(y1),int(x1)), 4, colors[i], -1)
		img4 = cv2.circle(img4, (int(y2),int(x2)), 4, colors[i], -1)

	cv2.imshow('Img1_Depth_Markers', img3)
	cv2.imshow('Img2_Depth_Markers', img4)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	if save_flag:
		cv2.imwrite('Img1_Depth_Markers.png', img3)
		cv2.imwrite('Img2_Depth_Markers.png', img4)

	for i in range(len(X)):
		print('Depth of ' + str(names[i]) + ' feature point = ' + str(X[i][2]))