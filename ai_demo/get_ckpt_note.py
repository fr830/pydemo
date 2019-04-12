import os
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

'''
版本1保存ckpt时，有两个文件model.ckpt-xxx（包含了参数名和参数值）和model.ckpt-xxx.meta（图结构）,要读取该ckpt时，路径按平常写法写

版本2保存模型时，有三个文件model.ckpt-xxx.data（参数值）、model.ckpt-xxx.index（参数名）、model.ckpt-xxx.meta（图结构）, 要读取该ckpt时，路径按只写三个文件的公共部分

'''

model_file = 'F:/demo/py/pydemo/ai_demo/vdsr/models/VDSR_adam_epoch_079.ckpt-248080.meta'
#
# checkpoint_path=os.path.join(model_file)
# reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map=reader.get_variable_to_shape_map()
#
# for key in var_to_shape_map:
#     print('tensor_name: ',key)

saver = tf.train.import_meta_graph(model_file)
with tf.Session() as sess:

    saver.restore(sess, 'F:/demo/py/pydemo/ai_demo/vdsr/models/VDSR_adam_epoch_079.ckpt-248080')

    graph_def = tf.get_default_graph().as_graph_def()

    print("node size : ", len(graph_def.node))
    for i, n in enumerate(graph_def.node):
        print("node name : %s --> %s" %(n.name, n.op))