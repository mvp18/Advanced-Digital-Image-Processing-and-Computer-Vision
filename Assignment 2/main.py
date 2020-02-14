import argparse
from modules import *

def main(args):

	# img = cv2.imread('chess_homo.png')
	
	if args.task_num==0:
		img = cv2.imread('Garden.JPG')
		task1(img)
	if args.task_num==1:
		img = cv2.imread('Garden.JPG')
		task2_3(img)
	if args.task_num==3:
		img = cv2.imread('Garden_cropped.jpg')
		task5(img)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Implementation of tasks in Assignment 2.")
	parser.add_argument('-task', '--task_num', help="0 - task 1 , 1 - tasks 2,3, 3 - task 5", default=1, type=int)

	args = parser.parse_args()
	print(args)
	main(args)