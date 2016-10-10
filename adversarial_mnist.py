import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils
from computational_graph import * # dump the computational graph into the global namespace to avoid boilerplate code for loading the model here. Mnist digits are also loaded.

"""The primary file used for generating adversarial images"""

def filter_mnist_by_digit(labels, filterdigit):
    """(np.array, int) -> [int]
    Return the indices corresponding to the value of filterdigit in the
    provided labels.

    :param labels: numpy array of shape (x, 10), each row containing a
    one-hot-encoded vector corresponding to the digit value of the associated
    mnist image. Assumes the encoding is zero-indexed.
    :param filterdigit: int corresponding to the digit to be filtered.
    """
    return [idx for idx,row in enumerate(labels) if row[filterdigit]==1]

def generate_labels(numrows, digit): 
    """(int, int) -> np.array
    Return a numpy array of shape (numrows, 10), each row containing a
    one-hot-encoded vector for the value specified by digit.

    Assumes a zero-indexed one-hot-encoding.

    :param numrows: int, the number of copies of the one-hot vector to include
    in the final array.
    :param digit: int, the index of the one-hot vector to turn on.
    """
    arr = np.zeros((numrows, 10))
    arr[:, digit] = 1
    return arr

def get_mnist_images_digit(tf_mnist, digit):
    """(tensorflow.contrib.learn.python.learn.datasets.base.Datasets, int) -> np.array
    Return all of the mnist examples of the specified digit from tf_mnist,
    across the train, validation, and test sets.

    :param tf_mnist: the tensorflow mnist datasets object (see typehint above)
    :param digit: the digit to retrieve
    """
    idx_train = filter_mnist_by_digit(mnist.train.labels, digit)
    train = mnist.train.images[idx_train]
    idx_test = filter_mnist_by_digit(mnist.test.labels, digit)
    test = mnist.test.images[idx_test]
    idx_val = filter_mnist_by_digit(mnist.validation.labels, digit)
    val = mnist.validation.images[idx_val]
    return np.vstack([train, test, val])

def gen_whitenoise_samps(shp):
    return tf.select(tf.random_uniform([shp,784], 
                     dtype=tf.float32) > 0.5, 
                     tf.ones([shp,784]), 
                     tf.zeros([shp,784]))

if __name__=='__main__':

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    six_label = generate_labels(1, 6)

    imgs = get_mnist_images_digit(mnist, 2)

    adx = utils.fgsm_tf(x, y_conv, 0.2)

    adversarial_twos = np.zeros((len(imgs), 784))
    niter = 0
    for img in imgs:
        feed_dict = {x: img.reshape(1,784), y_: six_label, keep_prob: 1.0}
        adv_ex = sess.run(adx, feed_dict=feed_dict)
        adversarial_twos[niter,:] += adv_ex[0]
        niter += 1

    adversarial_twos_pred = y_conv.eval({x: adversarial_twos, keep_prob: 1.0})

    np.save('original_twos.npy', imgs)
    np.save('adversarial_twos.npy', adversarial_twos)
    np.save('adversarial_twos_pred.npy', adversarial_twos_pred)
