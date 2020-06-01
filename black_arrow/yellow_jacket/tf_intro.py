import tensorflow as tf


@tf.function
def compute(a, b, c):
    d = a * b + c
    e = a * b * c
    return d, e

