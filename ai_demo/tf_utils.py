
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np;

# tf.argmax 计算某一维最大值的索引， 0 代表计算列  1 代表行
# tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素  相等为 True， 否则为 False
# tf.cast 布尔型转为dtype
# tf.reduce_mean 求平均数

def print_info():
    print(tf.__version__)
    print(tf.__path__)

    print(device_lib.list_local_devices())


def fun_conv2d():
    filter_weight = tf.get_variable('weight', [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))

    biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.1))

    input = tf.placeholder('input', [1, 256, 256, 3], dtype='float32')

    conv =tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding='SAME')

    conv = tf.nn.bias_add(conv, biases)

    conv = tf.nn.relu(conv)

from tensorflow.python.framework import graph_util


def tf_convert():
    with tf.Session() as sess:
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['inputs_raw', 'generator/generator_unit/output_stage/conv/Conv/BiasAdd'])
        with tf.gfile.FastGFile('./SRGAN_pre-trained/srgan.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())