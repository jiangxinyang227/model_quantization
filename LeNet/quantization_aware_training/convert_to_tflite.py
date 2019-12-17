"""
将存储了伪量化信息的freeze pb文件转换成完全量化的tflite文件，可以看见量化完之后文件内存基本减小到1/4
"""

import tensorflow as tf

path_to_frozen_graphdef_pb = 'pb_model/freeze_eval_graph.pb'
converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(path_to_frozen_graphdef_pb,
                                                              ["inputs"],
                                                              ["predictions"])

converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats = {"inputs": (0., 1.)}
converter.allow_custom_ops = True
converter.default_ranges_stats = (0, 255)
converter.post_training_quantize = True
tflite_model = converter.convert()
open("tflite_model/eval_graph.tflite", "wb").write(tflite_model)
