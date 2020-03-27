from utils import *
from modules import *
import argparse


def main():

	img = cv2.imread('/home/swagatam/cpp_test/wwsIP/practice/assignment4/IMG_6479.jpg')

	dom_pixels = display_dominant_colors(img)


if __name__ == '__main__':
	main()