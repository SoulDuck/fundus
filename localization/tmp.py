import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
mask_img=np.load('4152450_20101228_R.npy')
#img=Image.fromarray(img)
mask_img=plt.imsave('tmp.jpg',mask_img ,cmap='Greys')

gray= cv2.imread('tmp.jpg',0)
plt.imshow(gray, cmap='Greys')
plt.show()
thresh = cv2.threshold(gray, 1,255, cv2.THRESH_BINARY)[1]
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print contours
ori_img = Image.open('/Users/seongjungkim/data/detection/resize/4152450_20101228_R.png')
ori_img=np.asarray(ori_img)
ori_img=cv2.drawContours(ori_img, contours, -1, (0,255,0), 3)
plt.imshow(ori_img)
plt.show()
#contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


