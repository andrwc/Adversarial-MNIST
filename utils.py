import numpy as np
import tensorflow as tf

_EPSILON = 10e-8

# via https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py
def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

# via https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py
def categorical_crossentropy(output, target, from_logits=False):
    '''Categorical crossentropy between an output tensor
    and a target tensor, where the target is a tensor of the same
    shape as the output.
    '''
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output,
                                reduction_indices=len(output.get_shape()) - 1,
                                keep_dims=True)
        # manual computation of crossentropy
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        return - tf.reduce_sum(target * tf.log(output),
                               reduction_indices=len(output.get_shape()) - 1)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(output, target)

# via https://github.com/openai/cleverhans/blob/master/cleverhans/utils_tf.py
def tf_model_loss(y, model, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """
    if mean:
        # Return mean of the loss
        return tf.reduce_mean(categorical_crossentropy(y, model))
    else:
        # Return a vector with the loss per sample
        return categorical_crossentropy(y, model)

# via https://github.com/openai/cleverhans/blob/master/cleverhans/attacks.py
# altered slightly to avoid explicit dependence on utils_tf.py, as we've
# brought the relvant code into this module instead
def fgsm_tf(x, predictions, eps, clip_min=None, clip_max=None):
    """
    TensorFlow implementation of the Fast Gradient 
    Sign method. 
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :param eps: the epsilon (input variation parameter)
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    """

    # Compute loss
    y = tf.to_float(tf.equal(predictions, tf.reduce_max(predictions, 1, keep_dims=True)))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    loss = tf_model_loss(y, predictions, mean=False)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x

