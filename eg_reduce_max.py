import tensorflow as tf

with tf.Session() as sess:
    print(sess.run(tf.reduce_max([1, 2, 3, 4, 0, 8, 11, 2, 1])))
