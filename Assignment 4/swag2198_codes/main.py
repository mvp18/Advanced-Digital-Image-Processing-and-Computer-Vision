from utils import *
from modules import *
import argparse


def main():

	print('\n***** Starting Colorland Tour! *****\n')

	img1 = cv2.imread('IMG_6477.jpg')
	img2 = cv2.imread('IMG_6481.jpg')

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
		cv2.imshow('target', img2_cropped)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
	# cv2.destroyAllWindows()

	dom1, pixels1 = display_dominant_color(img1_cropped)
	dom2, pixels2 = display_dominant_color(img2_cropped)

	print('Displaying Dominant colors in src and target, press q to quit and proceed . . .')
	while True:
		cv2.imshow('src_dom', dom1)
		cv2.imshow('target_dom', dom2)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
	# cv2.destroyAllWindows()

	target_rec = transfer_dominant_colors(img1_cropped, img2_cropped)
	disp_and_quit(target_rec, 'source image rendered in target illuminant')

	print('\n***** Hope you enjoyed the tour in Colorland! *****\n')


if __name__ == '__main__':
	main()