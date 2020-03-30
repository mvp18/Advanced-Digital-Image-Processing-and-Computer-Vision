import argparse
from modules import *

def main(args):

	# img = cv2.imread('chess_homo.png')
	
	if args.task_num==0:
		img = cv2.imread('../Garden.JPG')
		task1(img)
	if args.task_num==1:
		img = cv2.imread('../Garden.JPG')
		save_img, _, _ = task2_3(img)
		cv2.imwrite('../Garden_with_vanishing_line.png', save_img)
		print('Image saved to disk!')
	if args.task_num==2:
		img = cv2.imread('../Garden.JPG')
		task4(img)
	if args.task_num==3:
		img = cv2.imread('../Garden_cropped.jpg')
		task5(img)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Implementation of tasks in Assignment 2.")
	parser.add_argument('-task', '--task_num', help="0 - task 1 , 1 - tasks 2 and 3, 2 - task4, 3 - task 5", default=1, type=int)

	args = parser.parse_args()
	print(args)
	main(args)