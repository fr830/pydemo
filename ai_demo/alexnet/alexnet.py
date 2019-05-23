import tensorflow as tf

class AlexNet(object):

    def __init__(self, x, num_classes, skip_layer, weights_path = 'DEFAULT'):

        self.X = x
        self.NUM_CLASSES = num_classes
        self.PROB_RATE = 0.5
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        self.create()

    def create(self):
        # 1st Layer: Conv(w ReLu) -> Pool -> Lrn
        with tf.variable_scope('layer1') as scope:
            '''
            input : [batch_size, 227, 227, 3]
            middle : [batch_size, 55, 55, 96]
            output : [batch_size, 27, 27, 96]
            '''
            conv1 = self.conv(self.X, 11, 11, 96, 4, 4, name='conv1', padding='VALID')      # (227-11)/4+1=55
            pool1 = self.max_pool(conv1, 3, 3, 2, 2, name='pool1', padding='VALID')         # (55-3)/2+1=27
            lrn1 = self.lrn(pool1, 2, 2e-05, 0.75, name='lrn1')

        # 2nd Layer: Conv(w ReLu) -> Pool -> Lrn with 2 groups
        with tf.variable_scope('layer2') as scope:
            '''
            input : [batch_size, 27, 27, 96]
            middle : [batch_size, 27, 27, 256]
            output : [batch_size, 13, 13, 96]
            '''
            conv2 = self.conv(lrn1, 5, 5, 256, 1, 1, padding='SAME', name='conv2', groups=2)    # 填充的像素数 P=2*((5-1)/2)=4   (27-5+4)/1+1=27
            pool2 = self.max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')          # (27-3)/2+1=13
            lrn2 = self.lrn(pool2, 2, 2e-05, 0.75, name='lrn2')

        # 3rd Layer: Conv (w ReLu)
        with tf.variable_scope('layer3') as scope:
            conv3 = self.conv(lrn2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv(w ReLu) splitted into two groups
        with tf.variable_scope('layer4') as scope:
            conv4 = self.conv(conv3, 3, 3, 384, 1, 1, name='conv4', groups=2)

        # 5th Layer: Conv(w ReLu) -> pool splitted into teo groups
        with tf.variable_scope('layer5') as scope:
            conv5 = self.conv(conv4, 3, 3, 256, 1, 1, name='conv5')
            pool5 = self.max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC(w ReLu) -> Dropout
        with tf.variable_scope('layer6') as scope:
            flattened = tf.reshape(pool5, [-1, 6*6*256])
            fc6 = self.fc(flattened, 6*6*256, 4096, name='fc6')
            dropout6 = self.dropout(fc6, self.PROB_RATE)

        # 7th layer: FC (w ReLu) -> Dropout
        with tf.variable_scope('layer7') as scope:
            fc7 = self.fc(dropout6, 4096, 4096, name='fc7')
            dropout7 = self.dropout(fc7, self.PROB_RATE)

        # 8th layer: FC and return unscaled activations(for tf.nn.softmax_cross_entropy_with_logits)
        with tf.variable_scope('layer8') as scope:
            self.fc8 = self.fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

    def conv(self, x, filter_height, filter_width, num_filter, stride_x, stride_y, name, padding='SAME', groups=1):

        input_channels = int(x.get_shape()[-1])/groups
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_x, stride_y, 1], padding=padding)

        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[filter_width, filter_height, input_channels, num_filter])
            biases = tf.get_variable('biases', shape=[num_filter])

            if groups == 1:
                conv = convolve(x, weights)
            else:
                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)

                output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
                conv = tf.concat(axis=3, values=output_groups)

            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

            relu = tf.nn.relu(bias, name=scope.name)

            return relu

    def max_pool(self, x, filter_width, filter_height, stride_x, stride_y, name, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, filter_width, filter_height, 1], strides=[1, stride_x, stride_y, 1], padding=padding, name=name)

    def lrn(self, x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha, beta = beta, bias = bias, name = name)

    def fc(self, x, num_in, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)

            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
            if relu == True:
                relu = tf.nn.relu(act)
                return relu
            else:
                return act

    def dropout(self, x, prob_rate):
        return tf.nn.dropout(x, rate=prob_rate)
