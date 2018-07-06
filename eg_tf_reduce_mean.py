import tensorflow as tf

x = tf.constant([[1, 2],
                 [3, 4]], dtype=tf.float32)

# 总体平均
y = tf.reduce_mean(x)
# 按列求平均
y_0 = tf.reduce_mean(x, axis=0)
# 按行求平均
y_1 = tf.reduce_mean(x, axis=1)

with tf.Session() as sess:
    print(sess.run(y))
    print(sess.run(y_0))
    print(sess.run(y_1))
