import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def isvalid(i, j, r, c):
    if i >= r or j >= c or i < 0 or j < 0:
        return 0
    return 1

def euc_dist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def gaussian1D(x, mean = 0, sigma = 1):
    num = np.exp(-(((x-mean)**2) / (2.0*sigma**2)))
    den = sigma * np.sqrt(2*np.pi)
    return num / den

def gaussian(m, n, sigma = 1):
    g = np.zeros((m,n))
    m = m // 2
    n = n // 2
    for i in range(-m,m+1):
        for j in range(-n,n+1):
            den = 2.0*np.pi*(sigma**2)
            num = np.exp(-(i**2 + j**2) / (2*(sigma**2)))
            g[i+m][j+n] = num / den
    return g

def dfs_visit(image, visited, i, j, count):

    r, c = image.shape
    visited[i][j] = 255
    for k in range(i-1, i+2):
        for l in range(j-1, j+2):
            if isvalid(k, l, r, c):
                if (visited[i][j]==0 and image[i][j]==255):
                    dfs_visit(image, visited, k, l, count)

    return visited
