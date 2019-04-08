from tensorflow.python.tools import inspect_checkpoint as chkp
import tensorflow as tf

saver = tf.train.import_meta_graph("./ade20k/model.ckpt-27150.meta", clear_devices=True)

#【敲黑板！】这里就是填写输出节点名称惹
output_nodes = ["xxx"]

with tf.Session(graph=tf.get_default_graph()) as sess:
    input_graph_def = sess.graph.as_graph_def()
    saver.restore(sess, "./ade20k/model.ckpt-27150")
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                    input_graph_def,
                                                                    output_nodes)
    with open("frozen_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())


