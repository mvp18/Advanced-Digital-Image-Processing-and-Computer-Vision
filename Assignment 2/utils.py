import numpy as np
import cv2

def find_intersection(line1, line2):
	
	m1 = (line1[1][0]-line1[0][0])/(line1[1][1]-line1[0][1])
	c1 = line1[0][0] - m1*line1[0][1]

	# print("Line equation: ")
	# print('y = ' + str(m1) + "x" + ["", "+"][c1 > 0] + str(c1))

	m2 = (line2[1][0]-line2[0][0])/(line2[1][1]-line2[0][1])
	c2 = line2[0][0] - m2*line2[0][1]

	# print("Line equation: ")
	# print('y = ' + str(m2) + "x" + ["", "+"][c2 > 0] + str(c2))

	x = (c2-c1)/(m1-m2)
	y = m1*x + c1

	return (x,y)