import argparse
from modules import *
from utils import *

def main(args):

	img1 = cv2.imread('Amitava_first.JPG')
	img2 = cv2.imread('Amitava_second.JPG')

	if args.descriptor_type:
		X1, X2 = compute_SURF_ckp(img1, img2, args.save_flag)	
	else:
		X1, X2 = compute_SIFT_ckp(img1, img2, args.save_flag)

	print('\nKeypoints in Image 1:\n{}'.format(X1))
	print('\nKeypoints in Image 2:\n{}'.format(X2))

	fundamental_matrix = solve_8pt_corr(X1, X2)

	print('\nFundamental Matrix:\n{}'.format(fundamental_matrix))

	e1, e_1 = draw_epipolar_lines(X1, X2, fundamental_matrix, img1, img2, args.save_flag)

	e2, e_2 = calc_epipoles_from_F(fundamental_matrix)

	print('\nDistance b/w 2 values for Left Epipole:{}'.format(np.sqrt((e1[0]-e2[0])**2 + (e1[1]-e2[1])**2)))
	print('Distance b/w 2 values for Right Epipole:{}'.format(np.sqrt((e_1[0]-e_2[0])**2 + (e_1[1]-e_2[1])**2)))

	P, P_ = estimate_proj_matrices(fundamental_matrix, e_2)

	#compatibility test
	S = np.matmul(np.matmul(P_.transpose(), fundamental_matrix), P)
	S[np.abs(S) < 1e-8] = 0 # limiting very small values to zero for skew-symmetric test
	S = np.around(S, decimals=6) # 2 values disagree in the 17th place of decimal, hence the limit to 6
	
	if (S.transpose() == -S).all(): # property of a skew-symmetric matrix
		print('Compatibility test passed!\n')

	print('Estimating scene point depths . . .\n')
	estimate_scene_depth(img1, img2, X1, X2, P, P_, args.save_flag)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Implementation of tasks in Assignment 3.")
	parser.add_argument('-dt', '--descriptor_type', help="0 - SIFT , 1 - SURF", default=0, type=int)
	# parser.add_argument('-mr', '--matcher', help="0 - Brute Force Matcher , 1 - FLANN based matcher", default=0, type=int)
	parser.add_argument('-save', '--save_flag', help="specify whether u want to save outputs(1) or just show them(0)", default=1, type=int)
	args = parser.parse_args()
	main(args)