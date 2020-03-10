import numpy as np
import cv2

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
	print(np.linalg.det(F_))

	return F_

def draw_epipolar_lines(X1, X2, F, img1, img2):

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

	cv2.imshow('Epipolar_lines1', img3)
	cv2.imshow('Epipolar_lines2', img4)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()