import tensorflow as tf
from alexnet import AlexNet  # import训练好的网络
import matplotlib.pyplot as plt

class_name = ['cat', 'dog']  # 自定义猫狗标签


def test_image(path_image, num_class, weights_path='Default'):
    # 把新图片进行转换
    img_string = tf.read_file(path_image)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    # img_decoded = tf.image.decode_jpeg(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_resized = tf.reshape(img_resized, shape=[1, 227, 227, 3])

    # 图片通过AlexNet
    model = AlexNet(img_resized, 0.5, 2, skip_layer='', weights_path=weights_path)
    score = tf.nn.softmax(model.fc8)
    max = tf.arg_max(score, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,
                      "D:/tensorflow/bvlc_alexnet/finetune_alexnet_with_tensorflow-master/checkpoints/model_epoch10.ckpt")  # 导入训练好的参数
        # score = model.fc8
        print(sess.run(model.fc8))
        prob = sess.run(max)[0]

        # 在matplotlib中观测分类结果
        plt.imshow(img_decoded.eval())
        plt.title("Class:" + class_name[prob])
        plt.show()


test_image('./test/20.png', num_class=2)  # 输入一张新图片