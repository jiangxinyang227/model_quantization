import re
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from tensorflow.contrib.slim import get_variables_to_restore
import config as cfg
from lenet import Lenet


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    sess = tf.Session()
    batch_size = cfg.BATCH_SIZE
    parameter_path = cfg.PARAMETER_FILE

    lenet = Lenet()

    max_iter = cfg.MAX_ITER

    variables = get_variables_to_restore()
    save_vars = [variable for variable in variables if not re.search("Adam", variable.name)]

    saver = tf.train.Saver(save_vars)

    sess.run(tf.initialize_all_variables())

    tf.summary.scalar("loss", lenet.loss)
    summary_op = tf.summary.merge_all()

    train_summary_writer = tf.summary.FileWriter("logs", sess.graph)

    for i in range(max_iter):
        batch = mnist.train.next_batch(batch_size)
        if i % 100 == 0:
            train_accuracy, summary = sess.run([lenet.train_accuracy, summary_op], feed_dict={
                lenet.raw_input_image: batch[0],
                lenet.raw_input_label: batch[1]
            })
            train_summary_writer.add_summary(summary)
            print("step %d, training accuracy %g" % (i, train_accuracy))

        if i % 500 == 0:
            test_accuracy = sess.run(lenet.train_accuracy, feed_dict={lenet.raw_input_image: test_images,
                                                                      lenet.raw_input_label: test_labels})
            print("\n")
            print("step %d, test accuracy %g" % (i, test_accuracy))
            print("\n")

        sess.run(lenet.train_op, feed_dict={lenet.raw_input_image: batch[0],
                                            lenet.raw_input_label: batch[1]})
    saver.save(sess, parameter_path)
    print("saved model")

    # 保存为saved_Model
    # Export checkpoint to SavedModel
    builder = tf.saved_model.builder.SavedModelBuilder("pb_model")
    inputs = {"inputs": tf.saved_model.utils.build_tensor_info(lenet.raw_input_image)}

    outputs = {"predictions": tf.saved_model.utils.build_tensor_info(lenet.predictions)}

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                  method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={"serving_default": prediction_signature},
                                         legacy_init_op=legacy_init_op,
                                         saver=saver)

    builder.save()


if __name__ == '__main__':
    main()
