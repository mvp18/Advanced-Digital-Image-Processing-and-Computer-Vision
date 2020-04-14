import argparse 
from utils import *
from modules import *

def main(args):

	range_img = cv2.imread('../RGBD_dataset/RGBD_dataset/'+args.range_img, 0)
	rgb_img = cv2.imread('../RGBD_dataset/RGBD_dataset/'+args.rgb_img)

	print('\nPress Esc to close image display windows at different stages. Turn on save_flag to also save them!')

	Ix = np.uint8(np.absolute(cv2.Sobel(range_img, cv2.CV_64F, 1, 0, ksize=-1)))
	Ixx = np.uint8(np.absolute(cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=-1)))
	
	Iy = np.uint8(np.absolute(cv2.Sobel(range_img, cv2.CV_64F, 0, 1, ksize=-1)))
	Iyy = np.uint8(np.absolute(cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=-1)))

	Ixy = np.uint8(np.absolute(cv2.Sobel(Iy, cv2.CV_64F, 1, 0, ksize=-1)))

	# cv2.imshow('Ix', range_img)
	# cv2.imshow('Ixx', Ix)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	H, K = mean_and_gaussian_curvatures(Ix, Iy, Ixx, Iyy, Ixy)
	K1, K2 = principal_curvatures(H, K)

	for i in range(range_img.shape[0]):
		for j in range(range_img.shape[1]):
			H = np.sign(H[i][j])
			K = np.sign(K[i][j])
			K1 = np.sign(K1[i][j])
			K2 = np.sign(K2[i][j])
			# print("H:{}, K:{}, K1:{}, K2:{}".format(H[i][j], K[i][j], K1[i][j], K2[i][j]))
			print('Point [{}][{}]; Topology by HK:{}; Topology by K1K2:{}'.format(i, j, topology_HK(H, K), topology_K2K1(K2, K1))

	find_NPS(range_img, args.DNP_thresh)

	# disp_2imgs(source_crop, source_dom, 'Source cropped', 'Dominant Color Whitened', args.save_flag, args.img1[:-4]+'_dominant_color.jpg')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Dominant Color Transfer between a pair of images")
	parser.add_argument('-range', '--range_img', help="specify source image name, image should be in image_folder directory", default='0.png', type=str)
	parser.add_argument('-rgb', '--rgb_img', help="specify target image name, image should be in image_folder directory", default='0.jpg', type=str)
	parser.add_argument('-th', '--DNP_thresh', help="minimum number of pts for accepting DNP into NPS", default=3, type=int)
	parser.add_argument('-save', '--save_flag', help="specify whether u want to save outputs(1) or just show them(0)", default=1, type=int)
	args = parser.parse_args()
	main(args)