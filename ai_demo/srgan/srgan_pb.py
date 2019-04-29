
import tensorflow as tf
from tensorflow.python.platform import gfile

import tensorlayer as tl
from PIL import Image
import numpy as np
import scipy.misc

pb_file = './models/srgan.pb'
image_lr_file = './images/img_001.png'

def get_images(filename, is_crop, fine_size, images_norm):
    img = scipy.misc.imread(filename, mode='RGB')
    if is_crop:
        size = img.shape
        start_h = int((size[0] - fine_size)/2)
        start_w = int((size[1] - fine_size)/2)
        img = img[start_h:start_h+fine_size, start_w:start_w+fine_size,:]
    img = np.array(img).astype(np.float32)
    if images_norm:
        img = (img-127.5)/127.5
    return img


def save_images(images, size, filename):
    return scipy.misc.imsave(filename, merge_images(images, size))


def merge_images(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((size[0] * h, size[1] * w, 3))

    for index, image in enumerate(images):
        i = index // size[1]
        j = index % size[0]
        imgs[i * h:i * h + h, j * w:j * w + w, :] = image

    return imgs


with tf.Session() as sess:
    with gfile.FastGFile(pb_file, 'rb') as f:  # 加载模型
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    input_tensor = sess.graph.get_tensor_by_name('input_target:0')
    output_tensor = sess.graph.get_tensor_by_name('generator/Tanh:0')

    batch_x = [get_images('./images/baboon.bmp', True, 256, True)]
    batchs = np.array(batch_x).astype(np.float32)

    results = sess.run(output_tensor, feed_dict={input_tensor: batchs})

    results = np.expand_dims(results[0], 0)
    save_images(results, [1, 1], './images/output.png')