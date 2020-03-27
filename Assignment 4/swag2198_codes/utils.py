from __future__ import print_function, division
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import KMeans

# Input to this is an RGB image (numpy array)
def img_to_rgb_matrix(img_rgb):
    
    m, n, c = img_rgb.shape
    img_rgb = img_rgb / 255
    # rgb_array is of shape 3 x no. of pixels and stores [R, G, B] value in each column
    rgb_array = np.array([img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]]).reshape(c, m*n)
    
    return rgb_array

# Input to this is a 3 x num_pixels shaped 2D matrix
def rgb_to_xyz(rgb_array):
    
    conv_mat = np.array([
        [0.6067, 0.1736, 0.2001],
        [0.2988, 0.5868, 0.1143],
        [0.0000, 0.0661, 1.1149]
    ])
    
    xyz_array = np.matmul(conv_mat, rgb_array)
    
    return xyz_array

# Takes an img which is read by opencv that is BGR, may be after cropping
# as required
def get_2D_chromatics(img):
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    rgb_array = img_to_rgb_matrix(img_rgb)
    xyz_array = rgb_to_xyz(rgb_array)
    
    xyz_array = xyz_array.transpose() #makes it [numpixels x 3] shaped
    for i in range(xyz_array.shape[0]):
        summ = np.sum(xyz_array[i])
        xyz_array[i][0] /= summ
        xyz_array[i][1] /= summ
    xy_array = xyz_array.transpose()[:-1,:] #It is [2 x numpixels] shaped
    
    return xy_array

def get_dominant_color(img_rgb, xy_array):
    
    m, n, c = img_rgb.shape
    
    data = pd.DataFrame({
        'x': xy_array[0],
        'y': xy_array[1]
    })
    
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(data)
    
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    
    k, counts = np.unique(labels, return_counts = True)
    mode = np.argmax(counts)
    pixels = counts[mode]
    
    indices = np.where(labels == mode)
    indices = indices[0]
    inds = []
    
    dom = img_rgb.copy()
    
    for i in range(len(indices)):
        x = indices[i] / n
        y = indices[i] % n
        dom[int(x), y] = np.array([255, 255, 255]) #color it white
        inds.append((int(x), y))
        
    dom = cv2.cvtColor(dom, cv2.COLOR_RGB2BGR)
    cv2.imwrite('dominant_color.jpg', dom)
    print('Dominant color image saved on disc! Please Check!')
    
    return inds # To be used later for color Transfer
