
import tensorflow as tf;
import numpy as np;

A = np.array([[1, 2, 3, 11],
			  [4, 5, 6, 12],
			  [7, 8, 9, 13]])

x = tf.transpose(A, [1 ,0])

B = np.array([[[1, 2 ,3],
			   [4 ,5 ,6]]])

y = tf.transpose(B, [2 ,1 ,0])

with tf.Session() as sess:
	print(A[1 ,0])
	print(sess.run(x))
	print(B[0 ,1 ,2])
	print(sess.run(y[2 ,1 ,0]))
