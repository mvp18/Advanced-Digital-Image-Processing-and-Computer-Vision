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

    print('Select any points on the image. Then the corressponding line will be displayed and its eqn in the terminal. Press Esc to exit.')
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
    print("Equation of displayed line: ")
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

    print('Select 2 sets of parallel lines (4 in total). This generates the vanishing line. A line parallel to it and passing through img centre will be displayed.')
    print('Further instructions follow!')
    #instantiate class
    clone = img.copy()
    line_pairs=[]
    
    for i in [0, 1]:
        if(i):
            pt_store = CoordinateStore((0, 0, 255))
            print('\nNow select 2nd parallel pair (4 points in order). Press Esc once done.')
        else:
            pt_store = CoordinateStore((0, 255, 0))
            print('\nSelect 1st parallel line pair (4 points in order). The 2 lines will be displayed once u select all 4. Press Esc once done.')
        
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
    
    img = clone.copy()
    cv2.line(img, (int(y1), int(x1)), (int(y2), int(x2)), (0, 255, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img, m_inf, c_inf
    

def task4(img):

    img_vline, m_inf, c_inf = task2_3(img)
    clone = img_vline.copy()

    class CoordinateStore:
        def __init__(self, color):
            self.point = ()
            self.color = color

        def select_point(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img_vline, (x,y), 4, self.color, -1)
                self.point = (x,y)
    
    #instantiate class
    pt_store = CoordinateStore((0, 0, 255))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', pt_store.select_point)

    print('Select any point on the green line in the image. Press c once done to break. Note that the last selected point will be used!')
    print('Three sets of parallel transformed lines will be shown passing through this point!')
    while(1):    
        cv2.imshow('image', img_vline)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("c"): # Press c to break
            break

    (k, h_) = pt_store.point
    h = (k - c_inf)/m_inf
    
    print(h, k)
    
    P = np.array([[h], [k], [1]])
    l2 = 1e-3
    l1 = (-1 - l2 * k) / h
    
    #First two rows of the matrix are random, third row is chosen to make
    #3rd coordinate = 0 as it will be transformed to an ideal point
    
    H = np.concatenate((np.random.rand(2, 3), np.transpose(np.array([[l1], [l2], [1]]))), axis=0)

    P_ = H.dot(P)
    
    #The point (h,k) gets mapped to (b,-a) which is also the slope of the parallel lines
    b, a = P_[0,0], -P_[1,0]
    
    #Choose two arbitrary c1 and c2 that correspond to ax + by + c1 = 0 and ax + by + c2 = 0 in real world
    
    #The lines arrays to be transformed back to image plane
    l=[]
    for i in range(6):
        c = (i+1)*np.random.rand()
        l.append(np.array([[a], [b], [c]]))
    
    #Matrix to transform real world coordinates to image plane coordinates
    Hinv = np.linalg.inv(H)
    
    #Since lines transform a/c to Hinverse here inverse(Hinv) = H
    Ht = H.transpose()
    li=[]
    for i in range(6):
        li.append(Ht.dot(l[i]))
    
    scale_arr = [400, -400, 200, -200, 100, -100]
    color_arr = [(255, 0, 255), (0, 255, 255), (0, 128, 255), (0, 0, 0), (153, 0, 76), (153, 0, 153)]
    img = clone.copy()

    for i in range(6):
        print("L" + str(i+1)+' :')
        print(str(li[i][0][0]) + 'x' + ["", "+"][li[i][1][0]>0] + str(li[i][1][0]) + "y" + ["", "+"][li[i][2][0]>0] + str(li[i][2][0]) + " = 0")

        mi = -li[i][1][0]/li[i][0][0]

        hi = h + scale_arr[i]
        ki = mi*scale_arr[i] + k

        cv2.line(img, (int(k), int(h)), (int(ki), int(hi)), color_arr[i], 2)
    
    cv2.circle(img, (int(k), int(h)), 4, (0, 0, 255), -1)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('../Garden_6_parallel_lines.png', img)
    print('Image saved to disk!')

def task5(img):

    print('Here u will need 2 sets of parallel lines similar to task 2_3. This will be needed for affine rectification on the cropped image.')
    print('Note we have cropped out only the ground portion of the garden for this task (as suggested in the assignment).')

    class CoordinateStore:
        def __init__(self, color):
            self.points = []
            self.color = color

        def select_point(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img, (x,y), 4, self.color, -1)
                self.points.append((x,y))

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
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
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

    pl1 = extract_lines(line_pairs[0], 0)
    pl2 = extract_lines(line_pairs[0], 2)
    pl3 = extract_lines(line_pairs[1], 0)
    pl4 = extract_lines(line_pairs[1], 2)
    
    img = clone.copy()
    corr = rectification_matrix(pl1, pl2, pl3, pl4)
    print(corr)
    res = homography(img, corr)

    cv2.imwrite('../garden_rect_color.png', res)
    print('Image saved to disk!')