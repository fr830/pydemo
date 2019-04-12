import tensorflow as tf
import numpy as np
import cv2
import scipy.io
from tensorflow.python.framework import graph_util

from ai_demo.vdsr.vdsr_model import model

ckpt_path = 'F:/demo/py/pydemo/ai_demo/vdsr/models/VDSR_adam_epoch_079.ckpt-248080'
pb_file = './models/vdsr.pb'
mat_file = 'F:/demo/py/pydemo/ai_demo/vdsr/images/0_2.mat'

with tf.Session() as sess:
    input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1), name='input')
    shared_model = tf.make_template('shared_model', model)
    output_tensor, weights = shared_model(input_tensor)
    saver = tf.train.Saver(weights)
    tf.global_variables_initializer().run()

    saver.restore(sess, ckpt_path)

    mat_dict = scipy.io.loadmat(mat_file)
    input_x = mat_dict["img_2"]

    vdsr_y = sess.run([output_tensor], feed_dict={input_tensor: np.resize(input_x, (1, input_x.shape[0], input_x.shape[1], 1))})
    vdsr_y = np.resize(vdsr_y, (input_x.shape[0], input_x.shape[1]))

    cv2.imshow("input_x", input_x)
    cv2.imshow("vdsr_y", vdsr_y)
    cv2.waitKey()

    # 转换为pb
    # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['input', 'shared_model/Add'])
    # with tf.gfile.FastGFile(pb_file, mode='wb') as f:
    #     f.write(constant_graph.SerializeToString())