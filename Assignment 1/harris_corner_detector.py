import cv2
import numpy as np

def calc_grad_x(img, k_sobel=3, norm=False):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, k_sobel)
    if norm:
        grad_x = cv2.normalize(grad_x, grad_x, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return grad_x

def calc_grad_y(img, k_sobel=3, norm=False):
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, k_sobel)
    if norm:
        grad_y = cv2.normalize(grad_y, grad_y, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return grad_y

def calc_grad_orientation(Ix, Iy):
    return np.arctan2(Iy, Ix)

def calc_grad_mag(Ix, Iy):
    return np.sqrt(Ix ** 2 + Iy ** 2)

def harris_values(img, window_size=5, harris_scoring=0.04, norm=False):
    # calculate image gradients on x and y dimensions
    Ix = calc_grad_x(img, 3)
    Iy = calc_grad_y(img, 3)
    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyx = Iy * Ix
    Iyy = Iy ** 2
    # create the weight window matrix
    # c = np.zeros((window_size,)*2, dtype=np.float32)
    # c[window_size/2, window_size/2] = 1.0
    # w = cv2.GaussianBlur(c, (window_size, window_size)*2, 0)
    w = np.ones((window_size,)*2)
    # calculate the harris values for all pixels of the image
    Rs = np.zeros(img.shape, dtype=np.float32)
    for r in range(w.shape[0]/2, img.shape[0] - w.shape[0]/2):
        minr = max(0, r - w.shape[0]/2)
        maxr = min(img.shape[0], minr + w.shape[0])
        for c in range(w.shape[1]/2, img.shape[1] - w.shape[1]/2):
            minc = max(0, c - w.shape[1]/2)
            maxc = min(img.shape[1], minc + w.shape[1])
            wIxx = Ixx[minr:maxr, minc:maxc]
            wIxy = Ixy[minr:maxr, minc:maxc]
            wIyx = Iyx[minr:maxr, minc:maxc]
            wIyy = Iyy[minr:maxr, minc:maxc]
            Mxx = (w * wIxx).sum()
            Mxy = (w * wIxy).sum()
            Myx = (w * wIyx).sum()
            Myy = (w * wIyy).sum()
            M = np.array([Mxx, Mxy, Myx, Myy]).reshape((2,2))
            Rs[r,c] = np.linalg.det(M)- harris_scoring * (M.trace() ** 2)
    if norm:
        Rs = cv2.normalize(Rs, Rs, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Rs

def harris_corners(img, window_size=5, harris_scoring=0.04, threshold=1e-2,
                  nms_size=10):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calculate harris values for all valid pixels
    Rs = harris_values(img, window_size, harris_scoring)
    # apply thresholding
    Rs = Rs * (Rs > (threshold * Rs.max())) * (Rs > 0)
    # apply non maximal suppression
    rows, columns = np.nonzero(Rs)
    new_Rs = np.zeros(Rs.shape)
    for r,c in zip(rows,columns):
        minr = max(0, r - nms_size / 2)
        maxr = min(img.shape[0], minr + nms_size)
        minc = max(0, c - nms_size / 2)
        maxc = min(img.shape[1], minc + nms_size)
        if Rs[r,c] == Rs[minr:maxr,minc:maxc].max():
            new_Rs[r,c] = Rs[r,c]
            #  Rs[minr:r, minc:c] = 0
            #  Rs[r+1:maxr, c+1:maxc] = 0
    return new_Rs

imgs = ['transA.jpg', 'transB.jpg', 'simA.jpg', 'simB.jpg']

# Harris Corners
# ==============
def ps4_1_a():
    images = imgs[0:3:2]
    for idx, img in enumerate(images):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        # calculate the X and Y gradients of the two images using the above filter
        img_grad_x = calc_grad_x(img, 3, norm=True)
        img_grad_y = calc_grad_y(img, 3, norm=True)
        # save the gradient pair
        cv2.imwrite('ps4-1-a-'+str(idx+1)+'.png', np.hstack((img_grad_x,
                                                                  img_grad_y)))
    print('Finished calculating and saved the gradients of the images!')
        #  cv2.imshow('', np.hstack((img_grad_x, img_grad_y)))
        # cv2.waitKey(0); cv2.destroyAllWindows()

def ps4_1_b():
    for idx, img_name in enumerate(imgs):
        # read image form file
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        # calculate harris values for image
        Rs = harris_values(img, window_size=3, harris_scoring=0.04, norm=True)
        # save harris values image
        cv2.imwrite('ps4-1-b-'+str(idx+1)+'.png', Rs)
    print('Finished and saved all harris value images to files.')



def ps4_1_c():
    for idx, img_name in enumerate(imgs):
        # read image form file
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        #  corners = cv2.cornerHarris(gray,2,3,0.06) > 0.001*corners.max()
        corners = harris_corners(img, window_size=3, harris_scoring=0.04,
                                 threshold=1e-3, nms_size=5)
        img[corners > 0] = [0, 0, 255]
        cv2.imwrite('ps4-1-c-'+str(idx+1)+'.png', img)
        #  cv2.imshow('',img); cv2.waitKey(0); cv2.destroyAllWindows()
    print('Finished harris corner detection and saved new images!')

imgs = ['simA.jpg']
ps4_1_a()
ps4_1_b()
ps4_1_c()
