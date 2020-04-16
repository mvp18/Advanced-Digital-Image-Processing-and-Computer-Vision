import argparse
import sys
from utils import *
from modules import *

sys.setrecursionlimit(10**8)

def main(args):

	range_img = cv2.imread('../RGBD_dataset/RGBD_dataset/'+args.range_img, 0)

	print('\nPress Esc to close image display windows at different stages. Turn on save_flag to also save them!\n')

	Ix = np.uint8(np.absolute(cv2.Sobel(range_img, cv2.CV_64F, 1, 0, ksize=-1)))
	Ixx = np.uint8(np.absolute(cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=-1)))
	
	Iy = np.uint8(np.absolute(cv2.Sobel(range_img, cv2.CV_64F, 0, 1, ksize=-1)))
	Iyy = np.uint8(np.absolute(cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=-1)))

	Ixy = np.uint8(np.absolute(cv2.Sobel(Iy, cv2.CV_64F, 1, 0, ksize=-1)))

	# cv2.imshow('Ix', range_img)
	# cv2.imshow('Ixx', Ix)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	print('Finding Curvatures and Topology per pixel.......\n')

	H, K = mean_and_gaussian_curvatures(Ix, Iy, Ixx, Iyy, Ixy)
	K1, K2 = principal_curvatures(H, K)

	HK_top = topology_HK(H, K)
	K1K2_top = topology_K2K1(K2, K1)

	if args.print_flag:
		for i in range(range_img.shape[0]):
			for j in range(range_img.shape[1]):
				print("H:{}, K:{}, K1:{}, K2:{}".format(H[i, j], K[i, j], K1[i, j], K2[i, j]))
				print('Point [{}][{}]; Topology by HK: {}; Topology by K1K2: {}'.format(i, j, HK_top[i][j], K1K2_top[i][j]))

	NPS_img = find_NPS(range_img, args.DNP_thresh, args.print_flag)

	seg_dict = {0:seg_NPS_bfs(NPS_img), 1:seg_gaussian_bfs(K), 2:seg_principal_bfs(K1K2_top)}
	savename_dict = {0:'img_'+args.range_img[0]+'_seg_NPS_thresh_'+str(args.DNP_thresh), 1:'img_'+args.range_img[0]+'_seg_gaussian', 
					 2:'img_'+args.range_img[0]+'_seg_principal'}

	print('Generating Segmented Range Image.......\n')
	
	imgseg = seg_dict[args.seg_method]

	disp_2imgs(range_img, imgseg, 'Original Range Img', 'Segmented Img', args.save_flag, savename_dict[args.seg_method]+'.png')

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description="Range Image Processing")
	
	parser.add_argument('-range', '--range_img', help="specify source image name, image should be in RGBD_dataset directory", default='0.png', type=str)
	parser.add_argument('-th', '--DNP_thresh', help="minimum number of pts for accepting DNP into NPS", default=3, type=int)
	parser.add_argument('-seg', '--seg_method', help="method of segmentation - NPS (0), gaussian curvature (1) and principal curvature(2)",
						default=0, type=int)
	parser.add_argument('-save', '--save_flag', help="specify whether u want to save outputs(1) or just show them(0)", default=0, type=int)
	parser.add_argument('-print', '--print_flag', help="specify whether u want to print curvatures, topology, NPS (1) or not (0)", default=0, type=int)
	
	args = parser.parse_args()
	main(args)