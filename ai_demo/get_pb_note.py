import tensorflow as tf
import os

model_dir = 'F:/ai/models/inception_v3_2016_08_28_frozen/'
model_name = 'inception_v3_2016_08_28_frozen.pb'

def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

create_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name,'\n')