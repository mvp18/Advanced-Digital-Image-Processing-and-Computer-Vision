from utils import *

def mean_and_gaussian_curvatures(Ix, Iy, Ixx, Iyy, Ixy):

	H = np.zeros((Ix.shape[0], Ix.shape[1]))
	K = np.zeros((Ix.shape[0], Ix.shape[1]))

	for i in range(Ix.shape[0]):
		for j in range(Ix.shape[1]):
			E = 1+Ix[i][j]**2
			F = Ix[i][j]*Iy[i][j]
			G = 1+Iy[i][j]**2
			den = np.sqrt(1+Ix[i][j]**2+Iy[i][j]**2)
			e = -Ixx[i][j]/den
			f = -Ixy[i][j]/den
			g = -Iyy[i][j]/den
			den = E*G-F**2
			H[i][j] = (E*g+G*e-2*F*f)/(2*den)
			K[i][j] = (e*g-f**2)/den

	return H, K

def principal_curvatures(H, K):

	K1 = np.zeros((H.shape[0], H.shape[1]))
	K2 = np.zeros((H.shape[0], H.shape[1]))

	for i in range(K1.shape[0]):
		for j in range(K1.shape[1]):
			sq = H[i][j]**2-K[i][j]
			if abs(sq)<=1e-14:
				sq=0.0
			K1[i][j] = H[i][j] + np.sqrt(sq)
			K2[i][j] = H[i][j] - np.sqrt(sq)

	return K1, K2

def topology_HK(H, K):

	Kminus = {-1:'saddle ridge', 0:'minimal surface', 1:'saddle valley'}
	Kzero = {-1:'ridge', 0:'flat', 1:'valley'}
	Kplus = {-1:'peak', 0:'none', 1:'pit'}
	curv_dict_KH = {-1:Kminus, 0:Kzero, 1:Kplus}

	topology_arr = [['' for j in range(H.shape[1])] for i in range(H.shape[0])]

	for i in range(H.shape[0]):
		for j in range(H.shape[1]):
			topology_arr[i][j] = curv_dict_KH[np.sign(K[i, j])][np.sign(H[i, j])]

	return topology_arr

def topology_K2K1(K2, K1):

	K1minus = {-1:'peak', 0:'ridge', 1:'saddle'}
	K1zero = {-1:'ridge', 0:'flat', 1:'valley'}
	K1plus = {-1:'saddle', 0:'valley', 1:'pit'}
	curv_dict_K1K2 = {-1:K1minus, 0:K1zero, 1:K1plus}

	topology_arr = [['' for j in range(K1.shape[1])] for i in range(K1.shape[0])]

	for i in range(K1.shape[0]):
		for j in range(K1.shape[1]):
			topology_arr[i][j] = curv_dict_K1K2[np.sign(K1[i, j])][np.sign(K2[i, j])]

	return topology_arr

def find_NPS(img, threshold, print_flag):

	print('\nFinding NPS per pixel.......\n')

	NPS_img = np.zeros((img.shape[0], img.shape[1]), dtype=int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			NPS_img[i, j] = NPS_pixel(img, (i, j), threshold, print_flag)

	return NPS_img

def seg_NPS_dfs(NPS_img):

	r, c = NPS_img.shape
	label = 0

	visited = np.zeros((r, c))
	out_img = np.zeros((r, c))

	# init_index = (random.randint(0, r), random.randint(0, c))

	# for i in range(r):
	# 	for j in range(c):
	loop_counter = 0
	while np.count_nonzero(visited)!=r*c:
		
		if loop_counter:
			[row, col] = np.where(visited==0)
			rand_idx = random.randint(0, len(row)-1)
			(i, j) = (row[rand_idx], col[rand_idx])
		else:
			(i, j) = (random.randint(0, r), random.randint(0, c))			
		
		if visited[i, j]==0:
			dfs_visit(NPS_img, visited, out_img, i, j, r, c, label)
			label+=1
			loop_counter+=1
	
	return out_img

def seg_NPS_bfs(NPS_img):

	r, c = NPS_img.shape
	label = 0

	visited = np.zeros((r, c))
	label_img = np.zeros((r, c), dtype=int)

	for i in range(r):
		for j in range(c):
			if visited[i, j]==0:
				bfs_visit(NPS_img, visited, label_img, i, j, r, c, label)
				label+=1

	out_img = generate_seg_img(label_img, r, c, 0.0001)

	return out_img

def seg_NPS_sequential(A):

	r, c = A.shape
	label = 0
	EQ = [0]

	label_img = np.zeros((r, c), dtype=int)
	out_img = np.zeros((r, c), dtype=np.uint8)

	label_img[0, 0] = label

	for j in range(1, c):
		
		if A[0, j]==A[0, j-1]:
			label_img[0, j] = label_img[0, j-1]
		else:
			label+=1
			label_img[0, j] = label
			EQ.append(label)

	for i in range(1, r):

		if A[i, 0]==A[i-1, 0]:
			label_img[i, 0] = label_img[i-1, 0]
		else:
			label+=1
			label_img[i, 0] = label
			EQ.append(label)
	
	# First pass
	for i in range(1, r):
		for j in range(1, c):
			
			left_ind = (i, j-1)
			top_ind = (i-1, j)

			if A[i, j]==A[left_ind] and A[i, j]!=A[top_ind]: label_img[i, j]=label_img[left_ind]
			if A[i, j]==A[top_ind] and A[i, j]!=A[left_ind]: label_img[i, j]=label_img[top_ind]
			
			if A[i, j]!=A[top_ind] and A[i, j]!=A[left_ind]:
				label+=1
				label_img[i, j]=label
				EQ.append(label)

			if A[i, j]==A[top_ind] and A[i, j]==A[left_ind] and label_img[left_ind]==label_img[top_ind]: label_img[i, j]=label_img[top_ind]
			
			if A[i, j]==A[top_ind] and A[i, j]==A[left_ind] and label_img[left_ind]!=label_img[top_ind]:
				min_label = min(label_img[left_ind], label_img[top_ind])
				max_label = max(label_img[left_ind], label_img[top_ind])
				label_img[i, j] = min_label
				# print(EQ)
				# print(max_label)
				# print(min_label)
				EQ[max_label] = min_label

	for i in range(len(EQ)):
		if EQ[i]!=i:
			j = i
			while(EQ[j]!=j):
				j = EQ[j]
			EQ[i]=j

	#Second pass
	for i in range(r):
		for j in range(c):
			pix_label = label_img[i, j]
			if EQ[pix_label]!=pix_label:
				label_img[i, j]=EQ[pix_label]

	out_img = generate_seg_img(label_img, r, c, 0.01)

	return out_img

def seg_gaussian_bfs(K):

	sign_arr = np.sign(K)

	r, c = sign_arr.shape
	label = 0

	visited = np.zeros((r, c))
	label_img = np.zeros((r, c), dtype=int)
	out_img = np.zeros((r, c), dtype=np.uint8)

	for i in range(r):
		for j in range(c):
			if visited[i, j]==0:
				bfs_visit(sign_arr, visited, label_img, i, j, r, c, label)
				label+=1

	out_img = generate_seg_img(label_img, r, c, 0.01)

	return out_img

def seg_principal_bfs(K1K2_Top):

	r, c = len(K1K2_Top), len(K1K2_Top[0])
	label = 0

	visited = np.zeros((r, c))
	label_img = np.zeros((r, c), dtype=int)
	out_img = np.zeros((r, c), dtype=np.uint8)

	for i in range(r):
		for j in range(c):
			if visited[i, j]==0:
				bfs_visit(K1K2_Top, visited, label_img, i, j, r, c, label)
				label+=1

	out_img = generate_seg_img(label_img, r, c, 0.01)

	return out_img

