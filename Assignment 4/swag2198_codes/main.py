from utils import *
from modules import *
import argparse


def main():

	img1 = cv2.imread('/home/swagatam/cpp_test/wwsIP/practice/assignment4/IMG_6479.jpg')
	img2 = cv2.imread('/home/swagatam/cpp_test/wwsIP/practice/assignment4/IMG_6477.jpg')

	print('Press q to quit')
	while True:
		cv2.imshow('source', img1)
		cv2.imshow('target', img2)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
	cv2.destroyAllWindows()

	print('Click on an image point with mouse and release on second corner point of rectangle . . .')
	print('After 2 corner points are displayed on terminal press q to quit and proceed')
	print('Cropping Image 1 . . .')
	img1_cropped = crop_region(img1)
	print('Cropping Image 2 . . .')
	img2_cropped = crop_region(img2)

	print('Displaying Cropped Images, press q to quit and proceed . . .')
	while True:
		cv2.imshow('src', img1_cropped)
		cv2.imshow('tg', img2_cropped)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
	cv2.destroyAllWindows()

	pixels = display_dominant_color(img1_cropped)


if __name__ == '__main__':
	main()