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
	indx = []
	indy = [] 

	for i in range(len(dom_indices)):
		x = dom_indices[i]//cols
		y = dom_indices[i]%cols
		indx.append(x)
		indy.append(y)
		clone[x, y] = [255, 255, 255]

	img_indices = [np.array(indx), np.array(indy)]

	return clone, img_indices

def transfer_dom_color(source_img, target_img, target_indices):

	rows, cols, channels = source_img.shape

	target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

	target_lab = bgr2l_alpha_beta(target_img[target_indices[0], target_indices[1], :], img_flag=False)
	source_lab = bgr2l_alpha_beta(source_img)

	lm_am_bm = modify_l_alpha_beta(source_lab, target_lab)

	img_bgr = l_alpha_beta2bgr(lm_am_bm, rows, cols, channels)

	cv2.imshow('Source Image in Target Illumination', img_bgr)
	cv2.waitKey(0)
	cv2.destroyAllWindows()