import cv2
import numpy as np

image_file = 'C:\\Users\\86529\\Desktop\\ssd.jpg'

img = cv2.imread(image_file)

img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

b, g, r = cv2.split(img)
y, cr, cb = cv2.split(img_ycrcb)

image_data = np.array(y)
print(image_data)

cv2.imshow("img",y)
cv2.waitKey()