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

# Input to this is a [3 x (mn)] shaped numpy array
def rgb_matrix_to_img_rgb(rgb, m, n):
    
    rgb = rgb * 255
    rgb = rgb.astype('uint8')
    
    rgb = rgb.reshape(3, m, n)
    img_rgb = np.dstack((rgb[0], rgb[1], rgb[2]))
    
    return img_rgb # Returns an RGB image of size [m x n x 3]

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
    xind = []
    yind = []
    
    dom = img_rgb.copy()
    
    # This loop might be unnecessary if I use rgb_matrix[indices] = np.array([1,1,1])
    for i in range(len(indices)):
        x = indices[i] / n
        y = indices[i] % n
        dom[int(x), y] = np.array([255, 255, 255]) #color it white
        xind.append(int(x))
        yind.append(y)
    
    inds = (np.array(xind), np.array(yind))
        
    dom_bgr = cv2.cvtColor(dom, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('dominant_color1.jpg', dom_bgr)
    
    return dom, inds # To be used later for color Transfer

# Takes an RGB image as input if flag is False (default)
def rgb_to_lab(img_rgb, arrayflag = False):
    
    # flag = True means a [3 x mn] shaped matrix is passed
    if arrayflag == True:
        rgb = img_rgb
    else:
        rgb = img_to_rgb_matrix(img_rgb)
    
    mat1 = np.array([
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444]
    ])
    
    lms = np.matmul(mat1, rgb)
    LMS = np.log(lms + 1e-5) # To avoid log(0) error
    
    mat2 = np.diag(np.array([1/np.sqrt(3), 1/np.sqrt(6), 1/np.sqrt(2)]))
    
    mat3 = np.array([
        [1, 1, 1],
        [1, 1, -2],
        [1, -1, 0]
    ])
    
    mat = np.matmul(mat2, mat3)
    
    lab = np.matmul(mat, LMS)
    
    return lab # Returns [3 x mn] shaped lab color matrix

def lab_to_rgb_img(lab, m, n):
    
    mat1 = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -2, 0]
    ])
    
    mat2 = np.diag(np.array([1/np.sqrt(3), 1/np.sqrt(6), 1/np.sqrt(2)]))
    
    mat = np.matmul(mat1, mat2)
    LMS = np.matmul(mat, lab)
    lms = np.exp(LMS)
    
    mat3 = np.array([
        [4.4679, -3.5873, 0.1193],
        [-1.2186, 2.3809, -0.1624],
        [0.0497, -0.2439, 1.2045]
    ])
    
    rgb = np.matmul(mat3, lms)
    img_rgb = rgb_matrix_to_img_rgb(rgb, m, n)
    
    return img_rgb

# Returns std dev of each row of [3 x mn] shaped matrix
def std_devs(lab):
    
    m, n = lab.shape
    mean = np.mean(lab, axis = 1, keepdims = True)
    lab = lab - mean
    lab = lab**2
    lab = lab / n
    sigma = np.sqrt(np.sum(lab, axis = 1, keepdims = True))
    
    return sigma # Shaped [3 x 1]

# Takes 2 RGB images as input
def dominant_color_transfer(source, target):
    
    labs = rgb_to_lab(source)
    
    # Since we are transferring the dominant color from target to source,
    # we only need pixels belonging to dominant color from target image.
    img1 = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
    xy_array = get_2D_chromatics(img1)
    dom, pixels = get_dominant_color(target, xy_array)
    
    labt = target[pixels]
    labt = labt / 255
    labt = labt.transpose()
    labt = rgb_to_lab(labt, arrayflag = True)
    
    
    means = np.mean(labs, axis = 1, keepdims = True)
    meant = np.mean(labt, axis = 1, keepdims = True)
    
    sigmas = std_devs(labs)
    sigmat = std_devs(labt)
    
    f = np.array([sigmat[i] / sigmas[i] for i in range(sigmas.shape[0])])
    
    lab_star = labs - means
    lab_ = f * lab_star
    labm = lab_ + meant
    
    m, n, c = source.shape
    # Reconstruct target image from transformed lab space
    target_rec = lab_to_rgb_img(labm, m, n)
    
    return target_rec # Returns an RGB image