import cv2
import matplotlib.pyplot as plt
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

	fundamental_matrix = solve_8pt_corr(X1, X2)

	draw_epipolar_lines(X1, X2, fundamental_matrix, img1, img2)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Implementation of tasks in Assignment 3.")
	parser.add_argument('-dt', '--descriptor_type', help="0 - SIFT , 1 - SURF", default=0, type=int)
	# parser.add_argument('-mr', '--matcher', help="0 - Brute Force Matcher , 1 - FLANN based matcher", default=0, type=int)
	parser.add_argument('-save', '--save_flag', help="specify whether u want to save outputs(1) or just show them(0)", default=1, type=int)
	args = parser.parse_args()
	main(args)