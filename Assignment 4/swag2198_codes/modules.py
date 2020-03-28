from utils import *

corners = []

def disp_and_quit(img, str = ['image']):

	print('Press q to quit and proceed . . .')

	while True:
		cv2.imshow(str, img)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

def click_and_crop(event, x, y, flags, img):

	global corners
	# img, corners = params

	if event == cv2.EVENT_LBUTTONDOWN:
		corners.append((x,y))
		print('Corner1: i = ' + str(y) + ", j = " + str(x))
	elif event == cv2.EVENT_LBUTTONUP:
		corners.append((x,y))
		print('Corner2: i = ' + str(y) + ", j = " + str(x))
		cv2.rectangle(img, corners[0], corners[1], (0,0,255), 2)

def crop_region(img):

	global corners
	clone = img.copy()
	cv2.namedWindow('image')
	cv2.setMouseCallback('image', click_and_crop)

	while True:
		cv2.imshow('image', img)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

	if len(corners) == 2:
		img_cropped = clone[corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]]
		print('Press q to quit and proceed . . .')
		while True:
			cv2.imshow('org', img)
			cv2.imshow('image', img_cropped)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
		# cv2.imshow('cropped', img_cropped)
		# cv2.waitKey(0)

	cv2.destroyAllWindows()
	corners = []
	return img_cropped


def display_dominant_color(img):

	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	xy_array = get_2D_chromatics(img)

	dom, pixels = get_dominant_color(img_rgb, xy_array)
	dom_bgr = cv2.cvtColor(dom, cv2.COLOR_RGB2BGR)

	print('Displaying Dominant Color . . .')
	print('Press q to quit from windows . . .')

	while True:
		cv2.imshow('original', img)
		cv2.imshow('dominant', dom_bgr)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

	cv2.destroyAllWindows()
	return pixels