import tensorflow as tf

# our NN's output
logits = tf.constant([[1, 2, 3],
                      [1, 2, 3],
                      [1, 2, 3]], dtype=tf.float32)

# step1: do softmax
# [[ 0.09003057  0.24472848  0.66524094]
#  [ 0.09003057  0.24472848  0.66524094]
#  [ 0.09003057  0.24472848  0.66524094]]
y = tf.nn.softmax(logits)
# true label
y_ = tf.constant([[0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 1]], dtype=tf.float32)
# step2: do log
# [[-2.40760589 -1.40760589 -0.40760601]
# [-2.40760589 -1.40760589 -0.40760601]
# [-2.40760589 -1.40760589 -0.40760598]]
y_log = tf.log(y)
# step3: do multiply
# [[-0.         -0.         -0.40760601]
#  [-0.         -0.         -0.40760601]
#  [-0.         -0.         -0.40760598]]
pixel_wise_mult = tf.multiply(y_, y_log)
# step4: do cross entropy
# 1.22282
cross_entropy = -tf.reduce_sum(pixel_wise_mult)
# 将标签稠密化
# [2 2 2]
dense_y = tf.argmax(y_, 1)

cross_entropy2_step1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=dense_y,
                                                                      logits=logits)
cross_entropy2_step2 = tf.reduce_sum(cross_entropy2_step1)

with tf.Session() as sess:
    y_value, y_log_value, pixel_wise_mult_value, cross_entropy_value, dense_y_value = sess.run(
        [y, y_log, pixel_wise_mult, cross_entropy, dense_y])
    sparse_cross_entropy2_step1_value, sparse_cross_entropy2_step2_value = sess.run(
        [cross_entropy2_step1, cross_entropy2_step2])
    print("step1:softmax result=\n%s\n" % (y_value))
    print("step2:y_log_result result=\n%s\n" % (y_log_value))
    print("step3:pixel_mult=\n%s\n" % (pixel_wise_mult_value))
    print("step4:cross_entropy result=\n%s\n" % (cross_entropy_value))
    print("Dense_y result =\n%s\n" % (dense_y_value))
    print("Function(softmax_cross_entropy_with_logits) result=\n%s\n" % (sparse_cross_entropy2_step1_value))
    print("Function(tf.reduce_sum) result=\n%s\n" % (sparse_cross_entropy2_step2_value))
