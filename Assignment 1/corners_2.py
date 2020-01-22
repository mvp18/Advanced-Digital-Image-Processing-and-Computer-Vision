import cv2
from scipy import signal as sig
from scipy import ndimage as ndi
from skimage.feature import corner_harris, corner_peaks
import numpy as np

def gradient_x(imggray):
	##Sobel operator kernels.
	kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
	return sig.convolve2d(imggray, kernel_x, mode='same')
def gradient_y(imggray):
	kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	return sig.convolve2d(imggray, kernel_y, mode='same')

img = cv2.imread('./images/checkerboard.jpg')
imggray = cv2.imread('./images/checkerboard.jpg', 0)

blur = cv2.GaussianBlur(imggray,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

I_x = gradient_x(th3)
I_y = gradient_y(th3)

Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
Iyy = ndi.gaussian_filter(I_y**2, sigma=1)

k = 0.05
detA = Ixx * Iyy - Ixy ** 2
traceA = Ixx + Iyy
harris_response = detA - k * traceA ** 2
corners = corner_peaks(harris_response)

img_copy_for_corners = np.copy(img)

# for rowindex, response in enumerate(corners):
# 	for colindex, r in enumerate(response):
# 		if r > 0:
# 			# this is a corner
# 			img_copy_for_corners[rowindex, colindex] = [0,0,0]

for corner in corners:
	x, y = corner[0], corner[1]
	img_copy_for_corners=cv2.circle(img_copy_for_corners, (x,y), 4, (0,0,255), -1)

cv2.imwrite("finalimage.png", img_copy_for_corners)