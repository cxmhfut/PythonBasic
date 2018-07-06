import tensorflow as tf

"""
tf.transpose(input,[dimension_1,dimension_2,...,dimension_n])
    交换张量的不同维度
    [2,1,0] 交换张量的第一维和第三维
"""

A = tf.constant([[1, 2, 3],
                 [4, 5, 6]], dtype=tf.float32)

x = tf.transpose(A, perm=[1, 0])

B = tf.constant([[[1, 2, 3],
                  [4, 5, 6]]], dtype=tf.float32)

y = tf.transpose(B, perm=[2, 1, 0])

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))