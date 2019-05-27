import os
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
from tensorflow.python.framework import graph_util

'''
版本1 保存ckpt时，有两个文件model.ckpt-xxx（包含了参数名和参数值）和model.ckpt-xxx.meta（图结构）,要读取该ckpt时，路径按平常写法写

版本2 保存模型时，有三个文件model.ckpt-xxx.data（参数值）、model.ckpt-xxx.index（参数名）、model.ckpt-xxx.meta（图结构）, 要读取该ckpt时，路径按只写三个文件的公共部分

'''

model_file = 'F:/snpe/snpe-1.24.0.256/models/mobilenet_ssd/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/model.ckpt.meta'

saver = tf.train.import_meta_graph(model_file)

with tf.Session() as sess:

    saver.restore(sess, 'F:/snpe/snpe-1.24.0.256/models/mobilenet_ssd/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/model.ckpt')

    graph_def = tf.get_default_graph().as_graph_def()

    print("node size : ", len(graph_def.node))
    for i, n in enumerate(graph_def.node):
        print("node name : %s --> %s" %(n.name, n.op))

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Preprocessor/sub',
                                                                                      'detection_boxes',
                                                                                      'detection_scores',
                                                                                      'detection_classes',
                                                                                      'num_detections'])
    with tf.gfile.FastGFile('./models/mobilenetssd.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())


# 已测试通过
# from tensorflow.python import pywrap_tensorflow
# import os
# checkpoint_path = 'F:/snpe/snpe-1.24.0.256/models/mobilenet_ssd/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/model.ckpt'
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print("tensor_name: ", key)