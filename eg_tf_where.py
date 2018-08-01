import tensorflow as tf

# 'input' has two true values, so output has two coordinates.
# 'input' has rank of 2, so coordinates have two indices.
input1 = tf.constant([[True, False],
                      [True, False]])
# input有2个True值，所以output的第一维度是2
# input有2个维度，所以output的第二维度是2
# output.shape = (2, 2)
# 第一个True的位置在[0 0]
# 第二个True的位置在[1 0]
# output = [[0 0]
#           [1 0]]

# 'input' has 5 true values, so output has 5 coordinates.
# 'input' has rank of 3, so coordinates have three indices.
input2 = tf.constant([[[True, False],
                       [True, False]],
                      [[False, True],
                       [False, True]],
                      [[False, False],
                       [False, True]]])
# input有5个True值，所以output的第一维度是5
# input有3个维度，所以output的第二维度是3
# output.shape = (5, 3)
# 第一个True的位置在[0 0 0]
# 第二个True的位置在[0 1 0]
# 第三个True的位置在[1 0 1]
# 第四个True的位置在[1 1 1]
# 第五个True的位置在[2 1 1]
# output = [[0 0 0]
#           [0 1 0]
#           [1 0 1]
#           [1 1 1]
#           [2 1 1]]

print(input1.shape)
print(input2.shape)

mask = tf.constant([True, False, True, False, True])
a = tf.constant([0, 0, 0, 0, 0])
b = tf.constant([1, 1, 1, 1, 1])
# mask和a，b具有相同的维度
# 将a中mask对应位置为True的位置保持不变，False的位置替换成b中对应的值

with tf.Session() as sess:
    print(sess.run(tf.where(input1)))
    print(sess.run(tf.where(input2)))
    print(sess.run(tf.where(mask, a, b)))
