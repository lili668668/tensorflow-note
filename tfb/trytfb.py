import tensorflow as tf
import numpy as np

# create input data and train data
xd = np.random.rand(100).astype(np.float32)
yd = xd * 0.1 + 0.3

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, name='x_input')
    ys = tf.placeholder(tf.float32, name='y_input')
    inputs = {'x_input': tf.saved_model.utils.build_tensor_info(xs)}

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
        outputs = {'y_output': tf.saved_model.utils.build_tensor_info(y)}

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y-ys))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
builder = tf.saved_model.builder.SavedModelBuilder("gs://learn_talk/demo17/output/")

# run the tensorflow
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)

    for step in range(2017):
        sess.run(train, feed_dict={xs: xd, ys: yd})
        if step % 20 == 0:
            result = sess.run(merged, feed_dict={xs: xd, ys: yd})
            writer.add_summary(result, step)
            print step, sess.run(Weights), sess.run(biases)
    feed_dict = {xs: [10, 20, 40]}
    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs, method_name='tensorflow/serving/predict')
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={'serving_default': signature})
    prediction = sess.run(y, feed_dict)
    print prediction
builder.save()

