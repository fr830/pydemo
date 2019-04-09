
import tensorflow as tf
import numpy as np

data = np.array([5, 3, 1, 6, 4, 2])

A = [[1,3,4,5,6]]
B = [[1,3,4,5,2]]

equ = tf.equal(A, B)

cas = tf.cast(equ, tf.float32)

mea = tf.reduce_mean(cas)

with tf.Session() as sess:
    print(sess.run(mea))

n = 0