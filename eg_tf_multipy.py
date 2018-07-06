import tensorflow as tf

a = tf.constant(2,shape=[4,5])
b = tf.constant(3,shape=[4,5])
c = tf.multiply(a,b)
print(a.shape)
print(b.shape)
print(c.shape)