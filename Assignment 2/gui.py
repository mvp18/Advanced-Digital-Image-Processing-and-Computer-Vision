import cv2
import numpy as np  

class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img,(x,y), 4, (0, 255, 0), -1)
            self.points.append((x,y))

#instantiate class
pt_store = CoordinateStore()
img = cv2.imread('Garden.JPG')
cv2.namedWindow('image')
cv2.setMouseCallback('image', pt_store.select_point)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord("c") or len(pt_store.points)==2:
        break

cv2.line(img, pt_store.points[0], pt_store.points[1], (0, 255, 0), 2)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Selected Coordinates: ")
for i in pt_store.points:
    print(i)


 