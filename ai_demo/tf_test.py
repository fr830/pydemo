
import tensorflow as tf

import tensorflow as tf
import numpy as np

pool = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

flat = tf.reshape(pool, [-1, 9])

with tf.Session() as sess:
    val = sess.run(flat)
    nn = 0