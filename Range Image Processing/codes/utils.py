import cv2
import numpy as np
import random

def isvalid(i, j, r, c):
	
	if i >= r or j >= c or i < 0 or j < 0:
		return 0
	return 1

def disp_2imgs(img1, img2, str1, str2, save_flag, save_name):

	combined_img = np.hstack((img1, img2))

	cv2.imshow(str1+' (Left) ' + str2 + ' (Right) ', combined_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	if save_flag:
		cv2.imwrite('../sample_results/'+save_name, combined_img)
		print('\nImage saved to disk!')

def NPS_pixel(img, pix, threshold, print_flag):

	neighborhood_3D = np.zeros((3, 3, 3))
	
	start_x=pix[0] if pix[0]-1<0 else pix[0]-1
	start_y=pix[1] if pix[1]-1<0 else pix[1]-1
	end_x = pix[0] if pix[0]+1>=img.shape[0] else pix[0]+1
	end_y = pix[1] if pix[1]+1>=img.shape[1] else pix[1]+1

	DNP_ids = [(i+1) for i in range(9)]
	pix_count = [0]*9

	DNP_dict = dict(zip(DNP_ids, pix_count))

	pt2plane = {(-1, 1, -1):[4, 7, 8], (0, 1, -1):[1, 7], (1, 1, -1):[5, 7, 9], (-1, 0, -1):[2, 8], (0, 0, -1):[1, 2, 4, 5],
				(1, 0, -1):[2, 9], (-1, -1, -1):[5, 6, 8], (0, -1, -1):[1, 6], (1, -1, -1):[4, 6, 9], (-1, 1, 0):[3, 4],
				(0, 1, 0):[1, 3, 8, 9], (1, 1, 0):[3, 5], (-1, 0, 0):[2, 3, 6, 7], (0, 0, 0):DNP_ids, (1, 0, 0):[2, 3, 6, 7],
				(-1, -1, 0):[3, 5], (0, -1, 0):[1, 3, 8, 9], (1, -1, 0):[3, 4], (-1, 1, 1):[4, 6, 9], (0, 1, 1):[1, 6],
				(1, 1, 1):[5, 6, 8], (-1, 0, 1):[2, 9], (0, 0, 1):[1, 2, 4, 5], (1, 0, 1):[2, 8], (-1, -1, 1):[5, 7, 9], 
				(0, -1, 1):[1, 7], (1, -1, 1):[4, 7, 8]}
	
	for i in range(start_x, end_x+1):
		for j in range(start_y, end_y+1):
			x = j-pix[1]
			y = pix[0]-i
			z = None
			if img[i, j]==img[pix[0], pix[1]]:
				z = 0
				neighborhood_3D[y, x, 1]=1
			elif img[i, j]==img[pix[0], pix[1]]-1:
				z = -1
				neighborhood_3D[y, x, 0]=1
			elif img[i, j]==img[pix[0], pix[1]]+1:
				z = 1
				neighborhood_3D[y, x, 2]=1
			if z is not None:
				for k in pt2plane[(x, y, z)]:
					DNP_dict[k]+=1

	NPS_planes = []
	NPS_bin = ['0']*9
	for plane in DNP_dict:
		if DNP_dict[plane]>threshold:
			NPS_bin[9-plane]='1'
			NPS_planes.append(plane)

	NPS_bin = "".join(NPS_bin)
	NPS_dec = int(NPS_bin, 2)

	if print_flag:
		print("[{}][{}] : {}".format(pix[0], pix[1], NPS_planes))

	return NPS_dec

def dfs_visit(A, visited, out_img, i, j, r, c, label):

	visited[i, j] = 1
	out_img[i, j] = label

	for k in range(i-1, i+2):
		for l in range(j-1, j+2):
			if isvalid(k, l, r, c):
				if (visited[k, l]==0 and A[k, l]==A[i, j]):
						dfs_visit(A, visited, out_img, k, l, r, c, label)

def bfs_visit(A, visited, label_img, i, j, r, c, label):

	queue = []
	visited[i, j] = 1
	label_img[i, j] = label
	queue.append((i, j))

	while queue:
		pix = queue.pop(0)
		for k in range(pix[0]-1, pix[0]+2):
			for l in range(pix[1]-1, pix[1]+2):
				if isvalid(k, l, r, c):
					if (visited[k, l]==0 and A[k][l]==A[pix[0]][pix[1]]):
							queue.append((k, l))
							visited[k, l] = 1
							label_img[k, l] = label

def generate_seg_img(label_img, r, c, smooth_thresh):

	out_img = np.zeros((r, c), dtype=np.uint8)

	unique_ele, count = np.unique(label_img, return_counts=True)
	unique_ele = list(unique_ele)
	label_with_max_count = unique_ele[np.argmax(count)]
	# sort_count_index = np.argsort(count)
	# count_arr_sorted = count[sort_count_index]

	for i in range(r):
		for j in range(c):
			pix_label = label_img[i, j]
			if count[unique_ele.index(pix_label)]<smooth_thresh*np.max(count):
				label_img[i, j] = label_with_max_count

	unique_ele, count = np.unique(label_img, return_counts=True)
	
	# print(unique_ele)
	print('\nNumber of connected components found : {}.'.format(len(unique_ele)))
	# print(count)

	sort_count_index = np.argsort(count)
	unique_ele = unique_ele[sort_count_index]
	# unique_ele = list(unique_ele)
	color = np.array([(255-i) for i in range(len(count))])
	color[color<0] = 0
	color[len(color)-1] = 0
	label2color = dict(zip(unique_ele, color))

	for i in range(r):
		for j in range(c):
			out_img[i, j] = label2color[label_img[i, j]]

	return out_img