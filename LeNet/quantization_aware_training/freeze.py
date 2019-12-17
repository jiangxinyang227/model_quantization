"""
将保存了训练时伪量化信息的checkpoint文件转换成freeze pb文件
"""
import tensorflow as tf
import config as cfg

from lenet import Lenet
from tensorflow.python.framework import graph_util

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


with tf.Session() as sess:
    le_net = Lenet(False)
    saver = tf.train.Saver()  # 不可以导入train graph，需要重新创建一个graph，然后将train graph图中的参数来填充该图
    saver.restore(sess, cfg.PARAMETER_FILE)

    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['predictions'])
    tf.io.write_graph(
        frozen_graph_def,
        "pb_model",
        "freeze_eval_graph.pb",
        as_text=False)
