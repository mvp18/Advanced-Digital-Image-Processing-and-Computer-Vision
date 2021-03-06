import cv2
import numpy as np
from sklearn.cluster import KMeans

def bgr_img2rgb_matrix(img):

	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	rows, cols, channels = img.shape

	rgb_mat = np.array([img[:,:,0], img[:,:,1], img[:,:,2]]).reshape(channels, rows*cols) # shape - (3, height*width)
	
	return rgb_mat

def bgr2xy(img):

	mat = np.array([
		[0.6067, 0.1736, 0.2001],
		[0.2988, 0.5868, 0.1143],
		[0.0000, 0.0661, 1.1149]
	])

	rgb_mat = bgr_img2rgb_matrix(img)
	
	img_XYZ = np.matmul(mat, rgb_mat)
	img_xyz = img_XYZ/img_XYZ.sum(axis=0, keepdims=True)
	img_xy = img_xyz[:2, :] # shape - (2, height*width)

	return img_xy

def disp_2imgs(img1, img2, str1, str2, save_flag, save_name):

	combined_img = np.hstack((img1, img2))

	cv2.imshow(str1+' (Left) ' + str2 + ' (Right) ', combined_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	if save_flag:
		cv2.imwrite('../sample_results/'+save_name, combined_img)

def bgr2l_alpha_beta(img, img_flag=True):

	if img_flag:
		rgb_mat = bgr_img2rgb_matrix(img)
	else:
		rgb_mat = img

	mat1 = np.array([
		[0.3811, 0.5783, 0.0402],
		[0.1967, 0.7244, 0.0782],
		[0.0241, 0.1288, 0.8444]
	])

	mat2 = np.diag(np.array([1/np.sqrt(3), 1/np.sqrt(6), 1/np.sqrt(2)]))
	
	mat3 = np.array([
		[1, 1, 1],
		[1, 1, -2],
		[1, -1, 0]
	])

	LMS = np.log(np.matmul(mat1, rgb_mat)+1e-6)

	l_alpha_beta = np.matmul(np.matmul(mat2, mat3), LMS)

	return l_alpha_beta

def modify_l_alpha_beta(source_lab, target_lab):

	lab_ = source_lab - np.mean(source_lab, axis=1, keepdims=True)

	lab_prime = lab_*np.std(target_lab, axis=1, keepdims=True)/np.std(source_lab, axis=1, keepdims=True)

	lm_am_bm = lab_prime + np.mean(target_lab, axis=1, keepdims=True)

	return lm_am_bm

def l_alpha_beta2bgr(lab, rows, cols, channels):

	mat1 = np.array([
		[1, 1, 1],
		[1, 1, -1],
		[1, -2, 0]
	])
	
	mat2 = np.diag(np.array([1/np.sqrt(3), 1/np.sqrt(6), 1/np.sqrt(2)]))

	mat3 = np.array([
		[4.4679, -3.5873, 0.1193],
		[-1.2186, 2.3809, -0.1624],
		[0.0497, -0.2439, 1.2045]
	])

	LMS = np.exp(np.matmul(np.matmul(mat1, mat2), lab))

	RGB = np.matmul(mat3, LMS)

	RGB = RGB.astype('uint8')

	RGB = RGB.reshape(channels, rows, cols)
	img_rgb = np.dstack((RGB[0], RGB[1], RGB[2]))

	img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

	return img_bgr




