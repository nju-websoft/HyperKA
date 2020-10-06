import tensorflow as tf
import numpy as np


PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0


# Real x, not vector!
def tf_atanh(x):
    return tf.atanh(tf.minimum(x, 1. - EPS))  # Only works for positive real x.


# Real x, not vector!
def tf_tanh(x):
    return tf.tanh(tf.minimum(tf.maximum(x, -MAX_TANH_ARG), MAX_TANH_ARG))


def tf_dot(x, y):
    return tf.reduce_sum(x * y, axis=1, keepdims=True)


def tf_norm(x):
    return tf.norm(x, ord=2, axis=-1, keepdims=True)