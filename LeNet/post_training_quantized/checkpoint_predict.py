import time
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


with tf.device("cpu:0"):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        gpu_options=gpu_options
    )

    sess = tf.Session(config=session_conf)

    saver = tf.train.import_meta_graph("checkpoint/variable.ckpt-100000.meta")
    saver.restore(sess, "checkpoint/variable.ckpt-100000")

    input_node = sess.graph.get_tensor_by_name('inputs:0')
    pred = sess.graph.get_tensor_by_name('predictions:0')

    start_time = time.time()

    labels = [label.index(1) for label in mnist.test.labels.tolist()]
    predictions = []
    for image in mnist.test.images:
        prediction = sess.run(pred, feed_dict={input_node: [image]}).tolist()[0]

        predictions.append(prediction)

    correct = 0
    for prediction, label in zip(predictions, labels):
        if prediction == label:
            correct += 1
    end_time = time.time()
    print((end_time - start_time) / len(labels) * 1000)
    print(correct / len(labels))