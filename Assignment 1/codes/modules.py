import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *

def convert_to_grayscale(image):
    B = image[:,:,0]
    G = image[:,:,1]
    R = image[:,:,2]
    img_gray = np.zeros(image.shape)
    img_gray = 0.2989*R + 0.5870*G + 0.1140*B
    img_gray = img_gray.astype('uint8')
    return img_gray

def filter2D(image, kernel):
    r, c = image.shape
    m, n = kernel.shape
    filtered = np.zeros(image.shape)
    dx, dy = m//2, n//2
    for i in range(r):
        for j in range(c):
            psum = 0.0
            for k in range(i-dx,i+dx+1):
                for l in range(j-dy,j+dy+1):
                    if isvalid(k,l,r,c):
                        psum += image[k][l] * kernel[i-k+dx][j-l+dy]
            filtered[i][j] = psum
    return filtered

def scaling(image, sigmag = 3, k = 5):
    kernel = gaussian(k,k,sigmag)
    imgn = image / 255.0
    scaled = filter2D(imgn,kernel)
    return scaled

def scaled_bilateral_filtering(image, sigmas = 4, sigmar = 12, sigmag = 3, k = 5):
    
    kernel = gaussian(k,k,sigmag)
    imgn = image / 255.0
    scaled = filter2D(imgn,kernel)
    
    r, c = imgn.shape
    filtered = np.zeros(image.shape)
    dx, dy = k//2, k//2
    
    for i in range(r):
        for j in range(c):
            numsum = 0.0
            densum = 0.0
            for k in range(i-dx,i+dx+1):
                for l in range(j-dy,j+dy+1):
                    if isvalid(k,l,r,c):
                        spatial = euc_dist(i,j,k,l)
                        rng = np.abs(imgn[k][l] - scaled[k][l])
                        Gs = gaussian1D(spatial,0,sigmas)
                        Gr = gaussian1D(rng,0,sigmar)
                        numsum += Gs*Gr*imgn[k][l]
                        densum += Gs*Gr
            filtered[i][j] = numsum / densum
    return filtered

def sharpen(image):
    laplacian = np.array(
    [
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]
    ])
    lap = filter2D(image, laplacian)
    lap = lap - np.min(lap)
    lap = lap * (255.0 / np.max(lap))
    
    sharpened = image + lap
    sharpened = sharpened - np.min(sharpened)
    sharpened = sharpened * (255.0 / np.max(sharpened))
    
    return sharpened

def edge_sobel(image):
    Sx = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])
    Sx = Sx / 8.0
    
    Sy = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ])
    Sy = Sy / 8.0
    
    Ix = filter2D(image, Sx)
    Iy = filter2D(image, Sy)
    
    grads = np.sqrt(Ix**2 + Iy**2)
    return grads

def connected_component(image):

    r, c = image.shape
    count=1

    visited = np.zeros([r, c])

    for i in range(r):
        for j in range(c):
            if isvalid(i, j, r, c):
                if (visited[i][j]==0 and image[i][j]==255):
                    visited = dfs_visit(image, visited, i, j, count)
                    count+=1
    return visited

def otsu(image):

    hist, _ = np.histogram(image, bins=256, range=(0, 255))
    total = image.shape[0]*image.shape[1]
    current_max, threshold = 0, 0
    sumT, sumF, sumB = 0, 0, 0
    
    for i in range(0,256):
        sumT += i * hist[i]
    
    weightB, weightF = 0, 0
    varBetween, meanB, meanF = 0, 0, 0
    
    for i in range(0,256):
        weightB += hist[i]
        weightF = total - weightB
        if weightF == 0:
            break
        sumB += i*hist[i]
        sumF = sumT - sumB
        meanB = sumB/weightB
        meanF = sumF/weightF
        varBetween = weightB * weightF
        varBetween *= (meanB-meanF)*(meanB-meanF)
        if varBetween > current_max:
            current_max = varBetween
            threshold = i  
    
    th = image
    th[th>=threshold]=255
    th[th<threshold]=0
    
    return th


