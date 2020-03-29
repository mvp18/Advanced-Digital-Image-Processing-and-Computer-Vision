import argparse 
from utils import *
from modules import *

def main(args):

	source_img = cv2.imread(args.img1)
	target_img = cv2.imread(args.img2)

	disp_2imgs(source_img, target_img, 'source', 'target')

	source_crop = click_and_crop(source_img)
	target_crop = click_and_crop(target_img)
	
	source_dom, source_indices = find_dominant_color(source_crop)
	target_dom, target_indices = find_dominant_color(target_crop)

	disp_2imgs(source_crop, source_dom, 'Source cropped', 'Dominant Color Whitened')
	disp_2imgs(target_crop, target_dom, 'Target cropped', 'Dominant Color Whitened')

	# transfer_dom_color(source_crop, target_crop, target_indices)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Dominant Color Transfer between a pair of images")
	parser.add_argument('-img1', '--img1', help="specify source image name, image should be in same directory", default='IMG_6477.jpg', type=str)
	parser.add_argument('-img2', '--img2', help="specify target image name, image should be in same directory", default='IMG_6479.jpg', type=str)
	parser.add_argument('-save', '--save_flag', help="specify whether u want to save outputs(1) or just show them(0)", default=1, type=int)
	args = parser.parse_args()
	main(args)