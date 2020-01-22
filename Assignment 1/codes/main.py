import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from utils import *
from modules import *

imgcv = cv2.imread('../images/cavepainting1.JPG')

print('Available Operations:\n')

op_dict = {'0':'Convert to grayscale', 
		   '1':'Scaled Bilateral Filtering', 
		   '2':'Sharpening', 
		   '3':'Sobel Edge Detector',
		   '4':'Otsu Thresholding',
		   '5':'Find connected components',
		   '6':'Erosion',
		   '7':'Dilation',
		   '8':'Opening',
		   '9':'Closing',
		   '10':'Harris Corner Point Detector'}

for k, v in op_dict.items():
	print(k, v)

while(1):

	operation = input('\nEnter operation you wish to perform:')
	operation = int(operation)

	def show_or_print(img_name, img, cmap):
		ans = input('\nSave(0) or Show(1) image? ')
		if ans=='0':
			save_dir = '../images'
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			plt.imsave(save_dir+'/'+img_name, np.uint8(img), cmap=cmap)
			print('\nImage saved! Check images folder.')
		else:
			plt.axis("off")
			plt.imshow(np.uint8(img), cmap=cmap)
			plt.show()

	if operation==0:
		img_gray = convert_to_grayscale(imgcv)
		show_or_print('grayscale.png', img_gray, cmap='gray')

	if operation==1:
		img_gray = convert_to_grayscale(imgcv)
		filtered = scaled_bilateral_filtering(img_gray, sigmas = 4, sigmar = 12, sigmag = 3, k = 5)
		show_or_print('bilateralfiltered(4,12,3,5x5).png', filtered, cmap='gray')

	if operation==2:
		img_gray = convert_to_grayscale(imgcv)
		sharpened = sharpen(img_gray)
		show_or_print('sharpened.png', sharpened, cmap='gray')

	if operation==3:
		img_gray = convert_to_grayscale(imgcv)
		sharpened = sharpen(img_gray)
		_, _, grads = edge_sobel(sharpened)
		show_or_print('sobeledges.png',grads, cmap='gray')

	if operation==4:
		img_gray = convert_to_grayscale(imgcv)
		sharpened = sharpen(img_gray)
		otsu_thresh = otsu(sharpened)
		show_or_print('otsu.png', otsu_thresh, cmap='gray')

	if operation==5:
		img_gray = convert_to_grayscale(imgcv)
		sharpened = sharpen(img_gray)
		otsu_thresh = otsu(sharpened)
		cc = connected_component(otsu_thresh)
		show_or_print('connected_component.png', cc, cmap='gray')

	if operation==6:
		img_gray = convert_to_grayscale(imgcv)
		sharpened = sharpen(img_gray)
		otsu_thresh = otsu(sharpened)
		eroded = erosion(otsu_thresh, 3)
		show_or_print('eroded.png', eroded, cmap='gray')

	if operation==7:
		img_gray = convert_to_grayscale(imgcv)
		sharpened = sharpen(img_gray)
		otsu_thresh = otsu(sharpened)
		dilated = dilation(otsu_thresh, 3)
		show_or_print('dilated.png', dilated, cmap='gray')

	if operation==8:
		img_gray = convert_to_grayscale(imgcv)
		sharpened = sharpen(img_gray)
		otsu_thresh = otsu(sharpened)
		opened = opening(otsu_thresh, 3)
		show_or_print('opened.png', opened, cmap='gray')

	if operation==9:
		img_gray = convert_to_grayscale(imgcv)
		sharpened = sharpen(img_gray)
		otsu_thresh = otsu(sharpened)
		closed = closing(otsu_thresh, 3)
		show_or_print('closed.png', closed, cmap='gray')

	if operation==10:
		harris_img = harris(imgcv, 1e-2, 10)
		show_or_print('harris.png', harris_img, cmap=None)

	ans = input('\nTry other operations (0) or exit (1)? ')
	if ans=='0':
		continue
	else:
		break