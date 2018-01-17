import numpy as np
import tensorflow as tf
def my_func(x):
  # x will be a numpy array with the contents of the placeholder below
  return np.sinh(x)
inp = tf.placeholder(tf.float32)
a=3
y = tf.py_func(my_func, [inp], tf.float32)

sess=tf.Session()
init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)

print sess.run(y , feed_dict={inp : 3.0})

import bbox_overlays
import bbox_transform

