import time
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
labels = [label.index(1) for label in mnist.test.labels.tolist()]
images = mnist.test.images

"""
预测的时候需要将输入归一化到标准正态分布
"""
means = np.mean(images, axis=1).reshape([10000, 1])
std = np.std(images, axis=1, ddof=1).reshape([10000, 1])
images = (images - means) / std
"""
需要将输入的值转换成uint8的类型才可以
"""
images = np.array(images, dtype="uint8")

interpreter = tf.contrib.lite.Interpreter(model_path="tflite_model/eval_graph.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

start_time = time.time()


predictions = []
for image in images:
    interpreter.set_tensor(input_details[0]['index'], [image])
    interpreter.invoke()
    score = interpreter.get_tensor(output_details[0]['index'])[0][0]
    predictions.append(score)

correct = 0
for prediction, label in zip(predictions, labels):
    if prediction == label:
        correct += 1
end_time = time.time()
print((end_time - start_time) / len(labels) * 1000)
print(correct / len(labels))
