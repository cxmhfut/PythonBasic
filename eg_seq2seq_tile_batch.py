import tensorflow as tf
from tensorflow.contrib import seq2seq
"""
tf.contrib.seq2seq.tile_batch([1, 2, 3], multiplier=5)
    
output:
    [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
"""
a = tf.constant([1, 2, 3])
output = seq2seq.tile_batch(a, multiplier=5)

with tf.Session() as sess:
    print(sess.run(output))
