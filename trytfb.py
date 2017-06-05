import tensorflow as tf
import numpy as np

# create input data and train data
xd = np.random.rand(100).astype(np.float32)
yd = xd * 0.1 + 0.3

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, name='x_input')
    ys = tf.placeholder(tf.float32, name='y_input')

# make a conputational graph
with tf.name_scope('layer'):
    with tf.name_scope('wieghts'):
        Weights = tf.Variable(tf.random_uniform([1], -1.0 , 1.0), name="W")
        tf.summary.histogram('weights', Weights)
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.zeros([1]), name="b")
        tf.summary.histogram('biases', biases)

    with tf.name_scope('Wx_add_b'):
        y = Weights * xs + biases

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y-ys))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# run the tensorflow
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for step in range(2017):
    sess.run(train, feed_dict={xs: xd, ys: yd})
    if step % 20 == 0:
        result = sess.run(merged, feed_dict={xs: xd, ys: yd})
        print step, sess.run(Weights), sess.run(biases)
feed_dict = {xs: [10, 20, 40]}
prediction = sess.run(y, feed_dict)
print prediction

