import tensorflow as tf
from tensorflow.contrib import seq2seq

a = tf.constant([1, 2, 3])
output = seq2seq.tile_batch(a, multiplier=5)

with tf.Session() as sess:
    print(sess.run(output))
