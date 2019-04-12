import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

# 加载mnist_inference.py 和mnist_train.py中定义的常量和函数。
from ai_demo.mnist import mnist_inference
from ai_demo.mnist import mnist_train
from PIL import Image
import numpy as np

pb_file = './model/mnist.pb'


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式。
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

        image_file = './images/0.jpg'
        image = Image.open(image_file)
        img_data = np.array(image).reshape(1, 784)

        images = mnist.validation.images[1]
        validate_feed = {x: img_data}

        # 直接通过调用封装好的函数来计算前向传播的结果。因为测试时不关注ze正则化损失的值
        # 所以这里用于计算正则化损失的函数被设置为None。
        y = mnist_inference.inference(x, None)

        correct_prediction = tf.argmax(y, 1, name="output")

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平均值了。这样就可以完全共用mnist_inference.py中定义的前向传播过程。
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名。
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型。
                saver.restore(sess, ckpt.model_checkpoint_path)

                ret = sess.run(correct_prediction, feed_dict=validate_feed)

                print('预测结果为 ： ', ret[0])

                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['x-input', 'y-input', 'output'])
                with tf.gfile.FastGFile(pb_file, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

            else:
                print("No checkpoint file found")
                return


def main(argv=None):
    mnist = input_data.read_data_sets("./data", one_hot=True)
    evaluate(mnist)


if __name__ == "__main__":
    tf.app.run()