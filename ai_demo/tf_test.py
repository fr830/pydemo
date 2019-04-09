
import tensorflow as tf

print(tf.__version__)
print(tf.__path__)

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# tf.argmax 计算某一维最大值的索引， 0 代表计算列  1 代表行
# tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素  相等为 True， 否则为 False
# tf.cast 布尔型转为dtype
# tf.reduce_mean 求平均数