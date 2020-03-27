from utils import *

def display_dominant_colors(img):

	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	xy_array = get_2D_chromatics(img)

	pixels = get_dominant_color(img_rgb, xy_array)

	return pixels