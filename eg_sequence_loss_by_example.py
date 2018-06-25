import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example

A = tf.random_normal([5, 4], dtype=tf.float32)
B = tf.constant([1, 2, 1, 3, 3], dtype=tf.int32)
W = tf.ones([5], dtype=tf.float32)

D = sequence_loss_by_example([A], [B], [W])

with tf.Session() as sess:
    print(sess.run(A))
    print(sess.run(B))
    print(sess.run(W))
    print(sess.run(D))
