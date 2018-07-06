import tensorflow as tf

"""
tf.expand_dims(tensor,dim)
"""
# a.shape = (3,)
# [ 1.  2.  3.]
a = tf.constant([1, 2, 3], dtype=tf.float32)
print('a.shape = ', a.shape)
# a_1.shape = (1,3)
# [[ 1.  2.  3.]]
a_0 = tf.expand_dims(a, axis=0)
print('a_0.shape = ', a_0.shape)
# y_2.shape = (3,1)
# [[ 1.]
#  [ 2.]
#  [ 3.]]
a_1 = tf.expand_dims(a, axis=1)
print('a_1.shape = ', a_1.shape)
# b.shape = (2,3)
b = tf.constant([[1, 2, 3],
                 [4, 5, 6]], dtype=tf.float32)
# b_0.shape = (1,2,3)
# [[[ 1.  2.  3.]
#   [ 4.  5.  6.]]]
b_0 = tf.expand_dims(b, axis=0)
print('b_0.shape = ', b_0.shape)
# b_1.shape = (2,1,3)
# [[[ 1.  2.  3.]]
#
#  [[ 4.  5.  6.]]]
b_1 = tf.expand_dims(b, axis=1)
print('b_1.shape = ', b_1.shape)
# b_2.shape = (2,3,1)
# [[[ 1.]
#   [ 2.]
#   [ 3.]]
#
#  [[ 4.]
#   [ 5.]
#   [ 6.]]]
b_2 = tf.expand_dims(b, axis=2)
print('b_2.shape = ', b_2.shape)

with tf.Session() as sess:
    a_0_value, a_1_value, b_0_value, b_1_value, b_2_value = sess.run(
        [a_0, a_1, b_0, b_1, b_2]
    )
    print(a_0_value)
    print(a_1_value)
    print(b_0_value)
    print(b_1_value)
    print(b_2_value)
