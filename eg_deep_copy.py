import tensorflow as tf
import copy

cell = tf.contrib.rnn.BasicLSTMCell(10)
cell1 = copy.deepcopy(cell)

a = tf.constant([1, 2, 3, 4, 5])
b = copy.deepcopy(a)
