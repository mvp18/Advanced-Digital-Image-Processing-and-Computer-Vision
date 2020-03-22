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
	print('Determinant of fundamental matrix : {}'.format(np.linalg.det(F_)))

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
	print('Right Epipole from F:{}'.format(e_))

	cv2.imshow('Epipolar_lines1', img3)
	cv2.imshow('Epipolar_lines2', img4)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return e, e_

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

	print('Camera Matrix P:\n{}'.format(P))
	print('Camera Matrix P`:\n{}'.format(P_))

	return P, P_