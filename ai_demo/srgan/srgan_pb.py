
import tensorflow as tf
from tensorflow.python.platform import gfile

import tensorlayer as tl
from PIL import Image
import numpy as np
import scipy.misc as sic

pb_file = './models/srgan.pb'

image_lr_file = './images/img_001.png'

with tf.Session() as sess:
    with gfile.FastGFile(pb_file, 'rb') as f:  # 加载模型
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    input_tensor = sess.graph.get_tensor_by_name('inputs_raw:0')
    output_tensor = sess.graph.get_tensor_by_name('encode_image/output_pngs/TensorArrayStack/TensorArrayGatherV3:0')

    image_lr_data = sic.imread(image_lr_file, mode="RGB").astype(np.float32)
    image_lr_data = image_lr_data / np.max(image_lr_data)
    input_im = np.array([image_lr_data]).astype(np.float32)

    results = sess.run(output_tensor, feed_dict={input_tensor: input_im})

    contents = results[0]
    with open('./images/img_001-outputs.png', "wb") as f:
        f.write(contents)