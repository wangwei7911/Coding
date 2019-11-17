import cv2
import numpy as np

img = cv2.imread('degree.jpg')
px = img[100, 100]
print(px)

blue = img[100, 100, 0]
print(blue)