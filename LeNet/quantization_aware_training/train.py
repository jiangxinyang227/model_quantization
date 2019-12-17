import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
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

    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())

    tf.summary.scalar("loss", lenet.loss)
    summary_op = tf.summary.merge_all()

    train_summary_writer = tf.summary.FileWriter("logs", sess.graph)

    for i in range(max_iter):
        batch = mnist.train.next_batch(batch_size)
        if i % 100 == 0:
            train_accuracy, summary = sess.run([lenet.train_accuracy, summary_op], feed_dict={
                lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]
            })
            train_summary_writer.add_summary(summary)
            print("step %d, training accuracy %g" % (i, train_accuracy))

        if i % 500 == 0:
            test_accuracy = sess.run(lenet.train_accuracy, feed_dict={lenet.raw_input_image: test_images,
                                                                      lenet.raw_input_label: test_labels})
            print("\n")
            print("step %d, test accuracy %g" % (i, test_accuracy))
            print("\n")

        sess.run(lenet.train_op, feed_dict={lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]})
    saver.save(sess, parameter_path)
    print("saved model")


if __name__ == '__main__':
    main()
