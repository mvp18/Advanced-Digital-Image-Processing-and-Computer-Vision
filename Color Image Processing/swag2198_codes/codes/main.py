from utils import *
from modules import *
import sys


def main():

	print('\n***** Starting Colorland Tour! *****\n')

	if len(sys.argv) != 3:
		print('Please provide source and target images!')
		return

	str1 = sys.argv[1]
	str2 = sys.argv[2]

	img1 = cv2.imread(str1)
	img2 = cv2.imread(str2)

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

	cv2.imwrite('source.png', img1_cropped)
	cv2.imwrite('target.png', img2_cropped)

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

	cv2.imwrite('source_dominant.png', dom1)
	cv2.imwrite('target_dominant.png', dom2)

	target_rec = transfer_dominant_colors(img1_cropped, img2_cropped)
	disp_and_quit(target_rec, 'source image rendered in target illuminant')

	cv2.imwrite('source_in_target_illuminant.png', target_rec)
	print('\n***** Hope you enjoyed the tour in Colorland! *****\n')


if __name__ == '__main__':
	main()