import tensorflow as tf
"""
求一个[start,end)的片段

a.shape = (3,2,3)

第一维度:start=0 end=2 strides=1 可以取0,1

[[[1, 1, 1], [2, 2, 2]],
 [[3, 3, 3], [4, 4, 4]]]
 
第二维度:start=0 end=2 strides=2 可以取0

[[[1, 1, 1]],
 [[3, 3, 3]]]
 
第三维度:start=0 end=2 strides=1 可以取0,1

[[[1, 1]],
 [[3, 3]]]

"""
a = [[[1, 1, 1], [2, 2, 2]],
     [[3, 3, 3], [4, 4, 4]],
     [[5, 5, 5], [6, 6, 6]]]

a = tf.constant(a)

print(a.shape)

b = tf.strided_slice(a, [0, 0, 0], [2, 2, 2], strides=[1, 2, 1])

with tf.Session() as sess:
    print(sess.run(b))

