import argparse 
from utils import *
from modules import *

def main(args):

	source_img = cv2.imread('../image_folder/'+args.img1)
	target_img = cv2.imread('../image_folder/'+args.img2)

	print('\nPress Esc to close image display windows at different stages. Turn on save_flag to also save them!')

	cv2.imshow('source', source_img)
	cv2.imshow('target', target_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	source_crop = click_and_crop(source_img)
	target_crop = click_and_crop(target_img)
	
	source_dom, source_indices = find_dominant_color(source_crop)
	target_dom, target_indices = find_dominant_color(target_crop)

	disp_2imgs(source_crop, source_dom, 'Source cropped', 'Dominant Color Whitened', args.save_flag, args.img1[:-4]+'_dominant_color.jpg')
	disp_2imgs(target_crop, target_dom, 'Target cropped', 'Dominant Color Whitened', args.save_flag, args.img2[:-4]+'_dominant_color.jpg')

	source_with_target_illmn = transfer_dom_color(source_crop, target_crop, target_indices)

	disp_2imgs(source_crop, source_with_target_illmn, 'Source Crop Original', 'With Target Illumination', args.save_flag,
		       args.img1[:-4]+'_x_'+args.img2[:-4]+'.jpg')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Dominant Color Transfer between a pair of images")
	parser.add_argument('-img1', '--img1', help="specify source image name, image should be in image_folder directory", default='IMG_6477.jpg', type=str)
	parser.add_argument('-img2', '--img2', help="specify target image name, image should be in image_folder directory", default='IMG_6479.jpg', type=str)
	parser.add_argument('-save', '--save_flag', help="specify whether u want to save outputs(1) or just show them(0)", default=1, type=int)
	args = parser.parse_args()
	main(args)