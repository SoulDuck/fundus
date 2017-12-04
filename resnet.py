#-*- coding:utf-8 -*-
import tensorflow as tf
from cnn import convolution2d, batch_norm_layer, affine, max_pool, avg_pool , gap

import cam

filters_per_blocks=[]
n_blocks=[]
a=3
class Resnet(object):
    def __init__ (self , x_ , phase_train ,  n_filters_per_box , n_blocks_per_box  , stride_per_box ,  use_bottlenect ,\
                  n_classes,activation=tf.nn.relu ,logit_type='gap'):
        """

        :param n_filters_per_box: [32, 64, 64, 128 , 256 ]  , type = list
        :param n_blocks_per_box:  [3, 5 , 4, 3, 2 ]  , type = list
        :param stride_per_box: [2, 2, 2, 2 , 2 ]  , type = list
        :param use_bottlenect: True , dtype = boolean
        :param activation:  , e.g_) relu
        :param logit_type: 'gap' or 'fc' , dtype = str
        """
        assert len(n_filters_per_box) == len(n_blocks_per_box) == len(stride_per_box)
        ### bottlenect setting  ###
        self.use_bottlenect = use_bottlenect
        self.activation = activation
        self.n_filters_per_box = n_filters_per_box
        self.n_blocks_per_box = n_blocks_per_box
        self.stride_per_box = stride_per_box
        self.n_boxes = len(n_filters_per_box)
        self.logit_type = logit_type
        self.x_ = x_
        self.phase_train = phase_train
        self.n_classes = n_classes
        """
        building model
        """
        self._build_model()

    def _build_model(self):
        with tf.variable_scope('stem'):
            # conv filters out = 64
            layer = convolution2d('conv_0', out_ch= 32,  x=self.x_, k=7, s=2)
            layer = batch_norm_layer(layer, phase_train= self.phase_train, scope_bn='bn_0')
            layer = self.activation(layer)
        for box_idx in range(self.n_boxes):
            print '#######   box_{}  ########'.format(box_idx)
            with tf.variable_scope('box_{}'.format(box_idx)):
                layer=self._box(layer , n_block= self.n_blocks_per_box[box_idx] , block_out_ch= self.n_filters_per_box[box_idx] ,
                          block_stride = self.stride_per_box[box_idx])
        self.logit=self._logit(layer ,  self.phase_train)



    def _box(self, x,n_block , block_out_ch , block_stride):
        """

        :param x:
        :param n_block: 5  , dtype = int
        :param block_out_ch: 32 , dtype = int
        :param block_stride: 2 , dtype = int
        :return:
        """
        layer=x
        for idx in range(n_block):
            if idx == n_block-1:
                layer = self._block(layer, block_out_ch=block_out_ch, block_stride=block_stride, block_n=idx)
                #box의 마지막 block에서는 이미지를 줄이기 위해서 주어진 strides 을 convolution 에 적용시킨다
            else:
                layer = self._block(layer , block_out_ch=block_out_ch , block_stride = 1 , block_n=idx)
        return layer
    def _block(self , x , block_out_ch  , block_stride  , block_n):
        shortcut_layer = x
        layer=x
        m=4 if self.use_bottlenect else 1
        out_ch = m * block_out_ch
        """ bottlenect layer """
        if self.use_bottlenect:
            with tf.variable_scope('bottlenect_{}'.format(block_n)):
                layer = batch_norm_layer(layer , self.phase_train  , 'bn_0')
                layer = convolution2d('conv_0' , layer , out_ch = block_out_ch , k =1 , s =1 ) #fixed padding padding = "SAME"
                layer = batch_norm_layer(layer, self.phase_train, 'bn_1')
                layer = convolution2d('conv_1', layer, out_ch=block_out_ch, k=3,
                                      s=block_stride)  # fixed padding padding = "SAME"
                layer = batch_norm_layer(layer, self.phase_train, 'bn_2')
                layer = convolution2d('conv_2', layer, out_ch=out_ch, k=1, s=1)  # fixed padding padding = "SAME"
                shortcut_layer = convolution2d('shortcut_layer', shortcut_layer, out_ch=out_ch, k=1, s=block_stride)
        else: #""" redisual layer """
            with tf.variable_scope('residual_{}.'.format(block_n)):
                layer = convolution2d('conv_0' , layer , block_out_ch , k=3 , s=block_stride) # in here , if not block_stride = 1 , decrease image size
                layer = batch_norm_layer(layer , self.phase_train,'bn_0' )
                layer = convolution2d('conv_1', layer, block_out_ch, k=3, s=1)
                shortcut_layer = convolution2d('shortcut_layer', shortcut_layer, out_ch=out_ch, k=1, s=block_stride)
        return shortcut_layer + layer


    def _logit(self ,x  , phase_train):
        if self.logit_type == 'gap':
            logit=gap('gap' , x , n_classes = self.n_classes)
        elif self.logit_type == 'fc':
            layer=affine('fc', x, out_ch=self.n_classes)
            logit = tf.cond(phase_train, lambda: tf.nn.dropout(layer, keep_prob=0.5), lambda: layer)
        else :
            print 'Not Implemneted , Sorry '
        logit=tf.identity(logit , 'logit')
        return logit

if __name__ =='__main__':
    phase_train = tf.placeholder(dtype=tf.bool , name='phase_train')
    x_ = tf.placeholder(dtype = tf.float32 , shape = [None , 32, 32 ,3 ] , name = 'x_')
    n_filters_per_box = [16,16,32,32]
    n_blocks_per_box = [5,5,5,5]
    stride_per_box= [5, 5, 5, 5]
    use_bottlenect = True
    model=Resnet(x_ , phase_train , n_filters_per_box , n_blocks_per_box , stride_per_box , \
                  use_bottlenect , n_classes=2 , activation=tf.nn.relu  , logit_type='gap' )
    print model.logit
