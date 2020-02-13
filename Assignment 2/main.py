import argparse
from modules import *

def main(args):
	
	img = cv2.imread('Garden.JPG')
	
	if args.task_num==0:
		task1(img)
	if args.task_num==1:
		task2_3(img)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Implementation of tasks in Assignment 2.")
	parser.add_argument('-task', '--task_num', help="0 - task 1 , 1 - tasks 2,3 ", default=1, type=int)

	args = parser.parse_args()
	print(args)
	main(args)