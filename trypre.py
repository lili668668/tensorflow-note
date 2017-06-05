import tensorflow as tf
import numpy as np

# create input data and train data
xd = np.random.rand(100).astype(np.float32)
yd = xd * 0.1 + 0.3

xs = tf.placeholder(tf.float32)
ys = tf.placeholder(tf.float32)

# make a conputational graph
Weights = tf.Variable(tf.random_uniform([1], -1.0 , 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * xs + biases

loss = tf.reduce_mean(tf.square(y-ys))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# run the tensorflow
sess = tf.Session()
sess.run(init)

for step in range(2017):
    sess.run(train, feed_dict={xs: xd, ys: yd})
    # if step % 20 == 0:
        # print step, sess.run(Weights), sess.run(biases)
feed_dict = {xs: [10, 20, 40]}
prediction = sess.run(y, feed_dict)
print prediction

