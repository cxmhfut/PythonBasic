import tensorflow as tf
"""
tf.nn.l2_loss(tensor)
    return sum(tensor**2)/2
"""
a = tf.constant([1,2,3],dtype=tf.float32)

with tf.Session() as sess:
    print(sess.run(tf.nn.l2_loss(a)))