import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import scipy.io
import cv2

pb_file = './models/vdsr.pb'
mat_file = 'F:/demo/py/pydemo/ai_demo/vdsr/images/0_2.mat'

with tf.Session() as sess:
    with gfile.FastGFile(pb_file, 'rb') as f:  # 加载模型
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    input = sess.graph.get_tensor_by_name('input:0')
    output = sess.graph.get_tensor_by_name('shared_model/Add:0')

    mat_dict = scipy.io.loadmat(mat_file)
    input_data = mat_dict["img_2"]

    vdsr = sess.run(output, feed_dict={input: np.resize(input_data, (1, input_data.shape[0], input_data.shape[1], 1))})
    vdsr = np.resize(vdsr, (input_data.shape[0], input_data.shape[1]))

    cv2.imshow("input", input_data)
    cv2.imshow("vdsr", vdsr)
    cv2.waitKey()