import tensorflow as tf

a = tf.constant([1, 3, 2])
b = tf.sequence_mask(a, 5)

"""
tf.sequence_mask([1, 3, 2], 5)

[[ True False False False False]
 [ True  True  True False False]
 [ True  True False False False]]

"""

with tf.Session() as sess:
    print(sess.run(b))
