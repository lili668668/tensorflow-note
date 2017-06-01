import tensorflow as tf
import numpy as np

# create input data and train data
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1, matrix2) # = np.dot(matrix1, matrix2)

# run the tensorflow
'''
sess = tf.Session()
result = sess.run(product)
print result
sess.close()
'''

with tf.Session() as sess:
    result = sess.run(product)
    print result



