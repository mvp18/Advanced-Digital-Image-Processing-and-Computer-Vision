import cv2
import numpy as np
from utils import *

def task1(img):
	
	class CoordinateStore:
	    def __init__(self):
	        self.points = []

	    def select_point(self, event, x, y, flags, param):
	        if event == cv2.EVENT_LBUTTONDOWN:
	            cv2.circle(img,(x,y), 4, (0, 255, 0), -1)
	            self.points.append((x,y))

	#instantiate class
	pt_store = CoordinateStore()
	cv2.namedWindow('image')
	cv2.setMouseCallback('image', pt_store.select_point)

	while(1):
	    cv2.imshow('image',img)
	    k = cv2.waitKey(20) & 0xFF
	    if k == ord("c") or len(pt_store.points)==2:
	        break

	cv2.line(img, pt_store.points[0], pt_store.points[1], (0, 255, 0), 2)
	cv2.imshow('image', img)

	print("Selected Coordinates: ")
	for i in pt_store.points:
	    print(i)
	
	slope = (pt_store.points[1][0]-pt_store.points[0][0])/(pt_store.points[1][1]-pt_store.points[0][1])
	intercept = pt_store.points[0][0] - slope*pt_store.points[0][1]
	print("Line equation: ")
	print('y = ' + str(slope) + "x" + ["", "+"][intercept > 0] + str(intercept))

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def task2_3(img):

	class CoordinateStore:
	    def __init__(self, color):
	        self.points = []
	        self.color = color

	    def select_point(self, event, x, y, flags, param):
	        if event == cv2.EVENT_LBUTTONDOWN:
	            cv2.circle(img, (x,y), 4, self.color, -1)
	            self.points.append((x,y))

	#instantiate class
	clone = img.copy()
	line_pairs=[]
	
	for i in [0, 1]:
		if(i):
			pt_store = CoordinateStore((0, 0, 255))
			print('\nNow select 2nd parallel pair (4 points in order). Press Esc once done.')
		else:
			pt_store = CoordinateStore((0, 255, 0))
			print('\nSelect 1st parallel line pair (4 points in order). Press Esc once done.')
		
		img = clone.copy()
		cv2.namedWindow('image')
		cv2.setMouseCallback('image', pt_store.select_point)
		
		while(1):
		    cv2.imshow('image',img)
		    k = cv2.waitKey(20) & 0xFF
		    if k == ord("c") or len(pt_store.points)==4:
		        break
		line_pairs.append(pt_store.points)
		if(i):
			cv2.line(img, pt_store.points[0], pt_store.points[1], (0, 0, 255), 2)
			cv2.line(img, pt_store.points[2], pt_store.points[3], (0, 0, 255), 2)
		else:
			cv2.line(img, pt_store.points[0], pt_store.points[1], (0, 255, 0), 2)
			cv2.line(img, pt_store.points[2], pt_store.points[3], (0, 255, 0), 2)

		cv2.imshow('image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	P1 = find_intersection([line_pairs[0][0], line_pairs[0][1]], [line_pairs[0][2], line_pairs[0][3]])
	P2 = find_intersection([line_pairs[1][0], line_pairs[1][1]], [line_pairs[1][2], line_pairs[1][3]]) 

	m_inf = (P2[1]-P1[1])/(P2[0]-P1[0])
	c_inf = P1[1] - m_inf*P1[0]
	print('\nEquation of vanishing line V:')
	print('y = ' + str(m_inf) + "x" + ["", "+"][c_inf > 0] + str(c_inf))
	
	h, w = img.shape[0], img.shape[1]
	x_c, y_c = h/2, w/2 # This (x, y) is in the proper (i, j) format
	
	print('\nEquation of line passing through centre and parallel to vanishing line:')
	print('y = ' + str(m_inf) + "x" + ["", "+"][y_c - m_inf*x_c > 0] + str(y_c - m_inf*x_c))

	y1 = 0
	x1 = x_c - (y_c/m_inf)

	y2 = w
	x2 = (w - y_c)/m_inf + x_c
	
	print(y1, x1)
	print(y2, x2)
	
	img = clone.copy()
	cv2.line(img, (int(y1), int(x1)), (int(y2), int(x2)), (0, 255, 0), 2)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()