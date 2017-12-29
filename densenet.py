import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cnn import convolution2d, batch_norm_layer, affine, max_pool, avg_pool , gap , dropout
import cam


def __init__(self, x_, phase_train, n_filters_per_box, n_blocks_per_box, stride_per_box, use_bottlenect, \
             n_classes, activation=tf.nn.relu, logit_type='gap', bottlenect_factor=4):
class Densenet(object):
    def __init__(self , phase_train , n_filters_per_box , n_blocks_per_box , stride_per_box , use_bottlenect , n_classes ,
                 activation=tf.nn.relu , logit_type='gap' , bottlenect_factor=4 ,**kwargs , keep_prob):

        self.phase_train = phase_train
        self.keep_prob = keep_prob

        _build_model()

        pass;

    #how to ? what is need? to make?
    # 인풋을 어디다가 설정하면 좋을까 fundus_densenet 에서 실행한다
    # 모델을 만든다

    def _build_model(self):
        pass;


    def _composite_function(self , _input , out_ch , kernel_size =3 ):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required

        input --> batch_norm --> relu --> convolution --> dropout
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = convolution2d(name='' , x_= output ,  out_ch=out_ch , k = kernel_size ,s = 1 , padding='SAME')
            # dropout(in case of training and in case it is no 1.0)
            output = dropout(output , self.phase_train , self.keep_prob)
        return output


    def conv2d(self , _input , out_features , kernel_size , strides=[1,1,1,1] , padding='SAME'):
        in_fearues=int(_input.get_shape()[-1])
        kernel=self.weight_variable_msra([kernel_size,kernel_size,in_fearues , out_features] , name='kernel')
        return tf.nn.conv2d(_input , kernel , strides , padding)




