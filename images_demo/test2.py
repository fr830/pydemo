import cv2
import numpy as np
import  struct
import scipy.io

'''
def rgb2ycbcr(img, only_y=True):
    # same as matlab rgb2ycbcr
    # only_y: only return Y channel
    # Input:
    #     uint8, [0, 255]
    #     float, [0, 1]
    
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

image_file = 'F:\\demo\\py\\pydemo\\data\\baboon.bmp'
img = cv2.imread(image_file)

y = rgb2ycbcr(img)
y = y.astype(np.float32)/255

height, weith = y.shape
y = y[0 : height - divmod(height, 12)[1], 0 : weith - divmod(weith, 12)[1]]

'''

mat_file = 'F:/demo/py/pydemo/ai_demo/vdsr/images/0_2.mat'
mat_dict = scipy.io.loadmat(mat_file)
input_data = mat_dict["img_2"]

input_data = input_data.astype('float32')
print(input_data.shape)
rows, cols = input_data.shape


with open('./files/0.raw','wb') as ff:
    for i in range(rows):
        for j in range(cols):
            bin_buf = struct.pack('f', input_data[i, j])
            ff.write(bin_buf)

with open('./files/0.raw','rb') as ff:

    ff.seek(0, 2)
    file_size = ff.tell()
    ff.seek(0, 0)

    fmt_size = file_size/4

    bin_buf = ff.read(file_size)

    data_raw = struct.unpack('f'*236160, bin_buf)

    nn = 0


