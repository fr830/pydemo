from PIL import Image
import numpy as np

import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("./data", one_hot=True)


# image_file = './images/3.jpg'
#
# image = Image.open(image_file)
#
# img_data = np.array(image)

# 拿对应的标签
# arr_data = mnist.train.labels[1]
# print(arr_data)  # one-hot形式

for n in range(10):
    im_data = np.array(np.reshape(mnist.validation.images[n], (28, 28)) * 255, dtype=np.int8)
    img = Image.fromarray(im_data, 'L')
    img_file = "./images/{0}.jpg".format(n)
    img.save(img_file)


