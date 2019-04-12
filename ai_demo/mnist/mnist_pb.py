import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np

model_file = 'F:\demo\\py\\pydemo\\ai_demo\\mnist\\model\\mnist.pb'

with tf.Session() as sess:
    with gfile.FastGFile(model_file, 'rb') as f:  # 加载模型
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    x = sess.graph.get_tensor_by_name('x-input:0')
    y = sess.graph.get_tensor_by_name('y-input:0')
    output = sess.graph.get_tensor_by_name('output:0')

    image_file = './images/0.jpg'
    image = Image.open(image_file)
    img_data = np.array(image, dtype='float32').reshape(1, 784)
    validate_feed = {x: img_data}

    ret = sess.run(output, feed_dict=validate_feed)
    print(ret)