

from PIL import Image
import numpy as np

image_file = './images/ssd.jpg'



im = Image.open(image_file)



print('Original image size: %sx%s' % (im.size))

im = im.convert("L")

matrix = np.asarray(im)             # 灰度图转矩阵
im2 = Image.fromarray(matrix)       # 矩阵转灰度图


im2.show()

# im = im.convert("L")
# data = im.getdata()
# data = np.matrix(data)
# #     print data
# # 变换成512*512
# data = np.reshape(data, (512, 512))
# new_im = Image.fromarray(data)
# # 显示图片
# new_im.show()