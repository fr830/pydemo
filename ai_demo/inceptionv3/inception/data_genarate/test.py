
import tensorflow as tf
from ai_demo.inceptionv3 import inception_v3

batch_size = 5
height, width = 299, 299
num_classes = 1000

inputs = tf.placeholder(tf.float32, [batch_size, height, width, 3], name='input')

logits, end_points = inception_v3.inception_v3(inputs, num_classes)

nn = 0