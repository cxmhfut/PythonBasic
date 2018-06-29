import tensorflow as tf
import numpy as np
"""
tf.nn.embedding_lookup(embedding, input_ids)

embedding:
    [[1 0 0 0 0]
     [0 1 0 0 0]
     [0 0 1 0 0]
     [0 0 0 1 0]
     [0 0 0 0 1]]
     
input_ids:
    [1 2 3 0 3 2 1]
    
output:
    [[0 1 0 0 0]
     [0 0 1 0 0]
     [0 0 0 1 0]
     [1 0 0 0 0]
     [0 0 0 1 0]
     [0 0 1 0 0]
     [0 1 0 0 0]]

"""
input_ids = tf.placeholder(tf.int32, [None])

embedding = tf.Variable(np.identity(5, np.int32))
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(embedding))
    print(sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 3, 2, 1]}))
