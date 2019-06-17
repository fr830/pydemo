
import tensorflow as tf;
import numpy as np;
import sys


class SquareTest(tf.test.TestCase):

    def testSquare1(self):
        with self.session():
            # 平方操作
            x = tf.square([2, 3])
            # 测试x的值是否等于[4,9]
            self.assertAllEqual(x.eval(), [4, 9])

    def testSquare2(self):
        with self.session():
            # 平方操作
            x = tf.square([2, 3])
            # 测试x的值是否等于[4,9]
            self.assertAllEqual(x.eval(), [4, 9])

if __name__ == "__main__":
    tf.test.main()