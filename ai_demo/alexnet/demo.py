
import tensorflow as tf
from alexnet import AlexNet

#
# train_file = 'D:/tensorflow/bvlc_alexnet/finetune_alexnet_with_tensorflow-master\train.txt'
# val_file = 'D:/tensorflow\bvlc_alexnet/finetune_alexnet_with_tensorflow-master\val.txt'
#
# with tf.device('/cpu:0'):
#     tr_data = ImageDataGenerator(filed,
#                                  train_file,
#                                  mode='training',
#                                  batch_size=batch_size,
#                                  num_classes=num_classes,
#                                  shuffle=True)
#
#     val_data = ImageDataGenerator(filed,
#                                   val_file,
#                                   mode='inference',
#                                   batch_size=batch_size,
#                                   num_classes=num_classes,
#                                   shuffle=False)
#
#     # create an reinitializable iterator given the dataset structure
#     iterator = Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
#     next_batch = iterator.get_next()
#
# x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3], name='input')
# y = tf.placeholder(tf.float32, [batch_size, num_classes], name='y')
# keep_prob = tf.placeholder(tf.float32)
#
# model = AlexNet(x, keep_prob, num_classes, train_layers)
#
# score = model.fc8
#
# var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
#
# with tf.name_scope("cross_ent"):
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))
#
# # Train op
# with tf.name_scope("train"):
#     # Get gradients of all trainable variables
#     gradients = tf.gradients(loss, var_list)
#     gradients = list(zip(gradients, var_list))
#
#     # Create optimizer and apply gradient descent to the trainable variables
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     train_op = optimizer.apply_gradients(grads_and_vars=gradients)
#
#
# with tf.name_scope("accuracy"):
#     correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for epoch in range(num_epochs):
#         img_batch, label_batch = sess.run(next_batch)
#         acc = sess.run(accuracy, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})

from datagenerator import *
import random
from datetime import datetime

# 加载数据
train_data, test_data = load_data()

# 定义网络超参数
learning_rate = 1e-4  # 学习率
training_epoches = 40  # 训练轮数
batch_size = 512  # 小批量大小
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
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=inference, labels=input_y))

with tf.name_scope('accuracy'):
    correct = tf.equal(tf.argmax(inference, 1), tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

saver = tf.train.Saver()

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

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

        # 测试
        test_batchs = [test_data[k:k + batch_size] for k in range(0, n_test, batch_size)]
        test_batchs.pop()
        test_accuracy_sum = []
        test_cost_sum = []
        for mini_batch in test_batchs:
            batch_x, batch_y = get_image_data_and_label(mini_batch, image_size=image_size, one_hot=True)
            test_accuracy, test_cost = sess.run([accuracy, loss], feed_dict={input_x: batch_x, input_y: batch_y})
            test_accuracy_sum.append(test_accuracy)
            test_cost_sum.append(test_cost)
        print('Epoch {0}: Test set accuracy {1}, loss {2}.'.format(i, np.mean(test_accuracy_sum), np.mean(test_cost_sum)))

        saver.save(sess, os.path.join('./models', 'alexnet'), global_step=len(train_batchs)*i)