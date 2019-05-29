
import tensorflow as tf
from alexnet import AlexNet

from datagenerator import *
import os, random
from datetime import datetime
import numpy as np

# 加载数据
train_data, test_data = load_data()

# 定义网络超参数
learning_rate = 1e-4  # 学习率
training_epoches = 40  # 训练轮数
batch_size = 256  # 小批量大小
num_classes = 10
train_layers = ['fc8', 'fc7', 'fc6']
n_train = len(train_data)  # 训练集数据长度
n_test = len(test_data)  # 测试卷数据长度
image_size = (227, 227)  # 图片大小


input_x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3], name='input_x')
input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name='input_y')

model = AlexNet(input_x, num_classes, train_layers)

inference = model.fc8

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=inference, labels=input_y))
    tf.summary.scalar('loss', loss)

with tf.name_scope('accuracy'):
    correct = tf.equal(tf.argmax(inference, 1), tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver()

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
    writer = tf.summary.FileWriter('./log', sess.graph)  # 将训练日志写入到logs文件夹下

    step = 0
    for i in range(training_epoches):
        random.shuffle(train_data)
        train_batchs = [train_data[k:k + batch_size] for k in range(0, n_train, batch_size)]
        train_batchs.pop()

        # 训练
        for j in range(len(train_batchs)):
            start_time = datetime.now()
            batch_x, batch_y = get_image_data_and_label(train_batchs[j], image_size=image_size, one_hot=True)
            sess.run(train_op, feed_dict={input_x : batch_x, input_y : batch_y})
            train_accuracy, train_loss = sess.run([accuracy, loss], feed_dict={input_x: batch_x, input_y: batch_y})
            use_time = (datetime.now()-start_time).seconds
            print('step {0}/{1} | {2}/{3} {4}s: training set accuracy : {5}, loss : {6}'.format(i, training_epoches, j, len(train_batchs), use_time, np.mean(train_accuracy), np.mean(train_loss)))

            result1 = sess.run(merged, feed_dict={input_x : batch_x, input_y : batch_y})
            writer.add_summary(result1, i*len(train_batchs) + j)

        # 测试
        test_batchs = [test_data[k:k + batch_size] for k in range(0, n_test, batch_size)]
        test_batchs.pop()
        test_accuracy_sum = []
        test_cost_sum = []
        for mini_batch in test_batchs:
            batch_x, batch_y = get_image_data_and_label(mini_batch, image_size=image_size, one_hot=True)
            test_accuracy, test_loss = sess.run([accuracy, loss], feed_dict={input_x: batch_x, input_y: batch_y})
            test_accuracy_sum.append(test_accuracy)
            test_cost_sum.append(test_loss)
        print('Epoch {0}: Test set accuracy {1}, loss {2}.'.format(i, np.mean(test_accuracy_sum), np.mean(test_cost_sum)))

        saver.save(sess, os.path.join('./models', 'alexnet'), global_step=len(train_batchs)*i)