import tensorflow as tf

"""
tf.multinomial(logits, num_samples, seed=None, name=None)
    logits: shape为[batch_size, num_classes]的2维tensor,每行[i,:]代表每类出现的概率
    num_samples: 独立采样数目
"""

x = tf.constant([[1, 1, 1]], dtype=tf.float32)  # 表示有3类，出现概率相等
# [[2 2 1 2 1 1 0 0 1 0]]
a = tf.multinomial(x, 10)

with tf.Session() as sess:
    a_value = sess.run(a)
    print(a_value)
