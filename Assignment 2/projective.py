from __future__ import division, print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_grid(grid_dim, block_dim):
    r = 2 ** grid_dim
    c = r
    br = 2 ** block_dim
    bc = br
    
    grid = np.zeros((r,c))
    color = 0
    for i in range(0,r,br):
        color = 255 - color
        for j in range(0,c,bc):
            grid[i:i+br-1, j:j+bc-1] = color
            color = 255 - color
    return grid

grid = create_grid(11,8)
plt.imsave('chess_aorg.png', grid, cmap = 'gray')

def isvalid(i, j, r, c):
    if i < 0 or j < 0 or i > r-1 or j > c-1:
        return 0
    return 1

def homography(img, matrix):
    res = np.zeros(img.shape)
    r, c = img.shape
    countval = 0
    countwhite = 0
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
                res[i1,j1] = img[i,j]
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

matrix = np.array([
    [1,-0.1,0],
    [-0.5,1,0],
    [0,0,1]
])
mat2 = np.array([
    [1,0,0],
    [0,1,0],
    [1e-4,1e-4,1]
])
mat3 = np.array([
    [1,0,0],
    [0,1,0],
    [-9.99*1e-5,-1e-5,1]
])
pl1 = (0.000001,0,0,909.091)
pl2 = (1696.421,0,1566.357,766.695)
pl3 = (0,0.00001,909.091,0)
pl4 = (0,1696.421,766.695,1566.357)
corr = rectification_matrix(pl1,pl2,pl3,pl4)
print(corr)

res2 = homography(grid,mat2)
res3 = homography(res2,corr)

plt.imshow(res2, cmap = 'gray')
plt.imsave('chess_homo.png', res2, cmap = 'gray')

plt.imshow(res3, cmap = 'gray')
plt.imsave('chess_homo_corrected.png', res3, cmap = 'gray')

def task4(point):
    h, k = point
    P = np.array([[h], [k], [1]])
    l2 = 1e-3
    l1 = (-1 - l2 * k) / h
    
    #First two rows of the matrix are random, third row is chosen to make
    #3rd coordinate = 0 as it will be transformed to an ideal point
    H = np.array([
        [1,2,3],
        [4,5,6],
        [l1,l2,1]
    ])
    P_ = matrix.dot(P)
    
    #The point (h,k) gets mapped to (b,-a) which is also the slope of the parallel lines
    b, a = P_[0,0], -P_[1,0]
    
    #Choose two arbitrary c1 and c2 that correspond to ax + by + c1 = 0 and ax + by + c2 = 0 in real world
    c1 = 100
    c2 = 100
    
    #The lines arrays to be transformed back to image plane
    l1 = np.array([[a], [b], [c1]])
    l2 = np.array([[a], [b], [c2]])
    
    #Matrix to transform real world coordinates to image plane coordinates
    Hinv = np.linalg.inv(mat)
    
    #Since lines transform a/c to Hinverse here inverse(Hinv) = H
    Ht = H.transpose()
    
    l1i = Ht.dot(l1)
    l2i = Ht.dot(l2)
    
    print("L1 :")
    
