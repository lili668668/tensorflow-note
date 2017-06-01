import tensorflow as tf
import numpy as np

def add_layer(inputs, inSize, outSize, af = None):
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))
    biases = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if af is None:
        outputs = Wx_plus_b
    else:
        outputs = af(Wx_plus_b)
    return outputs

xd = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, xd.shape)
yd = np.square(xd) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, af=tf.nn.relu)
prediction = add_layer(l1, 10, 1, af=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: xd, ys: yd})
        if i % 10 == 0:
            print sess.run(loss, feed_dict={xs: xd, ys: yd})
