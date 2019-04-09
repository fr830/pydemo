import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

ckpt_file = './tmp/model.ckpt'
ckpt_meta_file = './tmp/model.ckpt.meta'
pb_file = './tmp/model.pb'

# 生成模型并保存为ckpt
def test1():
    v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
    v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")

    init_op = tf.global_variables_initializer() # 初始化全部变量
    saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型

    with tf.Session() as sess:
        sess.run(init_op)
        print("v1:", sess.run(v1)) # 打印v1、v2的值一会读取之后对比
        print("v2:", sess.run(v2))
        saver_path = saver.save(sess, ckpt_file)
        print("Model saved in file:", saver_path)


# 定义图结构，加载ckpt data并运行
def test2():
    v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
    v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt_file)  # 即将固化到硬盘中的Session从保存路径再读取出来
        print("v1:", sess.run(v1))  # 打印v1、v2的值和之前的进行对比
        print("v2:", sess.run(v2))
        print("Model Restored")


# 加载图结构，加载ckpt data并运行
def test3():
    saver = tf.train.import_meta_graph(ckpt_meta_file)
    with tf.Session() as sess:
        saver.restore(sess, ckpt_file)
        # 通过张量的名称来获取张量
        v1 = tf.get_default_graph().get_tensor_by_name("v1:0")
        v2 = tf.get_default_graph().get_tensor_by_name("v2:0")
        print(sess.run(v1))
        print(sess.run(v2))


# 保存tf sess 为pb
def test4():
    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, name='x')
        y = tf.placeholder(tf.int32, name='y')
        b = tf.Variable(2, name='b')
        xy = tf.multiply(x, y)
        op = tf.add(xy, b, name='op_to_store')

        sess.run(tf.global_variables_initializer())
        print(sess.run(op, {x: 10, y: 3}))

        # convert_variables_to_constants 需要指定output_node_names，可以多个
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['x', 'y', 'op_to_store'])
        with tf.gfile.FastGFile(pb_file, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


# 加载pb 并运行
def test5():
    with tf.Session() as sess:
        with gfile.FastGFile(pb_file, 'rb') as f:  # 加载模型
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图

        #sess.run(tf.global_variables_initializer())
        #print(sess.run('b:0'))

        input_x = sess.graph.get_tensor_by_name('x:0')
        input_y = sess.graph.get_tensor_by_name('y:0')
        op = sess.graph.get_tensor_by_name('op_to_store:0')

        ret = sess.run(op, feed_dict={input_x: 6, input_y: 6})
        print(ret)


if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # test4()
    test5()