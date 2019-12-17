import re
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib.slim import get_variables_to_restore

model_dir = "checkpoint"
model_file = "checkpoint/variable.ckpt-50000"
new_model_file = "checkpoint/no_param_variable.ckpt-50000"

model_reader = pywrap_tensorflow.NewCheckpointReader(model_file)
var_dict = model_reader.get_variable_to_shape_map()
total_params = 0
quant_params = 0
for key in var_dict:
    total_params += 1
    if re.search("quant", key):
        quant_params += 1
    print(key)


print(total_params)
print(quant_params)

# 将bert中和adam相关的参数的值去掉，较小模型的内存
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    checkpoint_file = tf.train.latest_checkpoint(model_dir)
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)

    variables = get_variables_to_restore()
    other_vars = [variable for variable in variables if not re.search("Adam", variable.name)]
    # # print(other_vars)
    # for var in other_vars:
    #     if var.name == "bert/encoder/layer_0/intermediate/dense/weights_quant/min:0":
    #         print(var.name)
    #         print(sess.run(var))
    var_saver = tf.train.Saver(other_vars)
    var_saver.save(sess, new_model_file)
