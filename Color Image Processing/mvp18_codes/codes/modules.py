from utils import * 

def click_and_crop(img):

	class draw_rect:
		
		def __init__(self):
			
			self.points = []

		def select_points(self, event, x, y, flags, param):
			
			if event == cv2.EVENT_LBUTTONDOWN:
				self.points = [(x,y)]
			elif event == cv2.EVENT_LBUTTONUP:
				self.points.append((x, y))
				cv2.rectangle(img, self.points[0], self.points[1], (0, 255, 0), 2)

	clone = img.copy()
	rect = draw_rect()
	cv2.namedWindow('image')
	cv2.setMouseCallback('image', rect.select_points)

	print('\nPress mouse button down at 1st pt and release at 2nd to crop. Press c once done, r to do it again (if not done properly initially!)')

	while(1):
		cv2.imshow('image', img)
		k = cv2.waitKey(1) & 0xFF
		# if the 'r' key is pressed, reset the cropping region
		if k==ord("r"):
			img = clone.copy()
		# if the 'c' key is pressed, break from the loop
		elif k==ord("c"):
			break

	cv2.destroyAllWindows()
	
	if len(rect.points)==2:
		img_cropped = clone[rect.points[0][1]:rect.points[1][1], rect.points[0][0]:rect.points[1][0]]

	cv2.imshow('Original', img)
	cv2.imshow('Cropped', img_cropped)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return img_cropped

def find_dominant_color(img):

	clone = img.copy()

	xy_arr = bgr2xy(img)

	kmeans = KMeans(n_clusters=3)
	kmeans.fit(xy_arr.transpose())

	_, counts = np.unique(kmeans.labels_, return_counts=True)

	dom_indices = np.argwhere(kmeans.labels_==np.argmax(counts)).reshape(-1)

	rows, cols, channels = img.shape
	img_indices = []

	for i in range(len(dom_indices)):
		x = dom_indices[i]//cols
		y = dom_indices[i]%cols
		img_indices.append(dom_indices[i])
		clone[x, y] = [255, 255, 255] # whitens the pixels belonging to the cluster with the highest count

	# img_indices = tuple(img_indices)

	return clone, img_indices

def transfer_dom_color(source_img, target_img, target_indices):

	print('\nThis function renders the source image with illumination of target image.')

	rows, cols, channels = source_img.shape

	target_rgb_mat = bgr_img2rgb_matrix(target_img)
	target_lab = bgr2l_alpha_beta(target_rgb_mat[:, target_indices], img_flag=False)
	
	source_lab = bgr2l_alpha_beta(source_img)

	lm_am_bm = modify_l_alpha_beta(source_lab, target_lab)

	img_bgr = l_alpha_beta2bgr(lm_am_bm, rows, cols, channels)

	return img_bgr