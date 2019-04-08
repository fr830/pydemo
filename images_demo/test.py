import cv2
import numpy as np

'''
1、图像转换为矩阵

matrix = numpy.asarray(image)    
2、矩阵转换为图像

image = Image.fromarray(matrix)
'''

# 因为cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb) 与matlab中的rgb2ycbcr产生不同的结果，
# 并且SR算法一般采用rgb2ycbcr，所以模仿生成y
def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float64)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

image_file = 'F:\\demo\\py\\pydemo\\data\\baboon.bmp'

img = cv2.imread(image_file)

img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

b, g, r = cv2.split(img)
y = rgb2ycbcr(img)

y_ = y.astype(np.float64)/255

cv2.imshow("img",y_)
cv2.waitKey()

