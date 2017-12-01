import cnn
import tensorflow as tf
x_ = tf.placeholder(dtype = tf.float32 , shape=[None ,32 ,32 ,3 ])
y_ = tf.placeholder(dtype = tf.float32 , shape=[None , n_classes] )
cnn.algorithm()