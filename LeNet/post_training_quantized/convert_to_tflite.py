import tensorflow as tf

saved_model_dir = "./pb_model"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir,
                                                     input_arrays=["inputs"],
                                                     input_shapes={"inputs": [1, 784]},
                                                     output_arrays=["predictions"])
# converter.optimizations = ["DEFAULT"]  # 保存为v1,v2版本时使用
# converter.post_training_quantize = True  # 保存为v2版本时使用
tflite_model = converter.convert()
open("tflite_model_v3/eval_graph.tflite", "wb").write(tflite_model)
