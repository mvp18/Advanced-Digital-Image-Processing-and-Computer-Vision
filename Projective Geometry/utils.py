import numpy as np
import cv2

def extract_lines(line_pair, pair_num):

	(y1, x1), (y2, x2) = line_pair[pair_num], line_pair[pair_num+1]
	line = (x1, y1, x2, y2)

	return line

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

def isvalid(i, j, r, c):
    if i < 0 or j < 0 or i > r-1 or j > c-1:
        return 0
    return 1

def homography(img, matrix):
    res = np.zeros(img.shape)
    r, c, n = img.shape
    for i in range(r):
        for j in range(c):
            x = np.array([[i], [j], [1]])
            x_ = matrix.dot(x)
            i1, j1, k = x_[0,0], x_[1,0], x_[2,0]
            if k == 0:
                k = 1e-10
            i1 /= k
            j1 /= k
            i1, j1 = int(i1), int(j1)
            if isvalid(i1,j1,r,c):
                res[i1,j1, :] = img[i,j, :]
    return res

def rectification_matrix(pl1, pl2, pl3, pl4):
    #for 1st pair of parallel lines
    x1, y1, x2, y2 = pl1
    x3, y3, x4, y4 = pl2
    
    m1 = (y2 - y1) / (x2 - x1)
    c1 = y1 - m1*x1
    m2 = (y4 - y3) / (x4 - x3)
    c2 = y3 - m2*x3
    
    #(p1x,p1y) is one vanishing point
    p1x = (c2 - c1) / (m1 - m2)
    p1y = (m2*c1 - m1*c2) / (m2 - m1)
    
    #for 2nd pair of parallel lines
    x1, y1, x2, y2 = pl3
    x3, y3, x4, y4 = pl4
    
    m1 = (y2 - y1) / (x2 - x1)
    c1 = y1 - m1*x1
    m2 = (y4 - y3) / (x4 - x3)
    c2 = y3 - m2*x3
    
    #(p2x,p2y) is another vanishing point
    p2x = (c2 - c1) / (m1 - m2)
    p2y = (m2*c1 - m1*c2) / (m2 - m1)
    
    #slope and intercept of vanishing line i.e., y = mv*x + cv
    mv = (p2y - p1y) / (p2x - p1x)
    cv = p1y - mv*p1x
    
    #Find l1, l2, l3(=1) for sending vanishing line to (0,0,1)
    l1 = mv / cv
    l2 = -1.0 / cv
    
    #The
    mat = np.array([
        [1,0,0],
        [0,1,0],
        [l1,l2,1]
    ])
    
    return mat