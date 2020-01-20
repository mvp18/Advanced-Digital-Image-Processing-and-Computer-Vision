import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *
from modules import *

imgcv = cv2.imread('/home/swagatam/cpp_test/wwsIP/practice/cavepainting1.JPG')

img_gray = convert_to_grayscale(imgcv)
plt.imsave('/home/swagatam/cpp_test/wwsIP/practice/images/grayscale.png', img_gray, cmap = 'gray')

scaled = scaling(img_gray, sigmag = 3, k = 5)
plt.imsave('/home/swagatam/cpp_test/wwsIP/practice/images/gaussianfiltered(3,5x5).png', scaled, cmap = 'gray')

filtered = scaled_bilateral_filtering(img_gray, sigmas = 4, sigmar = 12, sigmag = 3, k = 5)
plt.imsave('/home/swagatam/cpp_test/wwsIP/practice/images/bilateralfiltered(4,12,3,5x5).png', filtered, cmap = 'gray')

sharpened = sharpen(img_gray)
plt.imsave('/home/swagatam/cpp_test/wwsIP/practice/images/sharpened.png', sharpened, cmap = 'gray')

grads = edge_sobel(img_gray)
plt.imsave('/home/swagatam/cpp_test/wwsIP/practice/images/sobeledges.png',grads, cmap = 'gray')

