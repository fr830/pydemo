import cv2
import numpy as np
import struct
import scipy.io


with open('./files/0g.raw', 'rb') as ff:
    ff.seek(0, 2)
    file_size = ff.tell()
    ff.seek(0, 0)

    fmt_size = file_size / 4

    bin_buf = ff.read(file_size)

    data_raw = struct.unpack('f' * 236160, bin_buf)

    data_tmp = np.array(data_raw, dtype='float32')

    data_tmp = data_tmp.reshape((480, 492))

    cv2.imshow("data_tmp", data_tmp)
    cv2.waitKey()

    nn = 0