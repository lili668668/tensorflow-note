import tensorflow as tf

def add_layer(inputs, inSize, outSize, af = None):
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))
    biases = tf.Variable(tf.zeros([1, outSize] + 0.1))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if af is None:
        outputs = Wx_plus_b
    else:
        outputs = af(Wx_plus_b)
    return outputs
