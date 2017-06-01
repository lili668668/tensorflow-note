import tensorflow as tf

state = tf.Variable(0, name='counter')
#print state.name

one = tf.constant(1)

newValue = tf.add(state, one)
update = tf.assign(state, newValue)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for cnt in range(3):
        sess.run(update)
        print sess.run(state)
