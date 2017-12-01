class A:
    def __init__(self):
        self.tmp_1()
    def tmp(self):
        print 'tmp'

    def tmp_1(self):
        self.tmp()

a=A()

import resnet

import tensorflow as tf
phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
x_ = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='x_')
n_filters_per_box = [16, 16, 32, 32]
n_blocks_per_box = [5, 5, 5, 5]
stride_per_box = [5, 5, 5, 5]
use_bottlenect = True
model = resnet.Resnet(x_, phase_train, n_filters_per_box, n_blocks_per_box, stride_per_box, \
               use_bottlenect, n_classes=2, activation=tf.nn.relu, logit_type='gap')
print model.logit
