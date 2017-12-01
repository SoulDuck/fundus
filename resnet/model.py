import tensorflow as tf
from cnn import convolution2d, batch_norm_layer, affine, max_pool, avg_pool , gap
import cam

filters_per_blocks=[]
n_blocks=[]

class Resnet():
    def __init__ (self , n_filters_per_box , n_blocks_per_box  , use_bottlenect , activation=tf.nn.relu , logit_type='gap'):
        assert len(n_filters_per_box) == len(n_blocks_per_box)
        ### bottlenect setting  ###
        self.use_bottlenect = use_bottlenect
        self.activation = activation
        self.n_filters_per_box = n_filters_per_box
        self.n_blocks_per_box = n_blocks_per_box
        self.n_boxes = len(n_filters_per_box)
        self.logit_type = logit_type
        """
        n_blocks_per_box
        """
    def _build_model(self):
        with tf.variable_scope('stem'):
            # conv filters out = 64
            layer = convolution2d('conv_0', x=x_, k=7, s=2)
            layer = batch_norm_layer(layer, train_phase=, scope_bn='bn_0')
            layer = self.activation(layer)
        for box_idx in range(self.n_boxes):
            with tf.variable_scope('box_{}'.foramt(box_idx)):
                self._box(layer , n_block= self.n_blocks_per_box[box_idx] , block_out_ch= self.n_filters_per_box[box_idx] )

    def _box(self, x,n_blocks , block_out_ch , block_stride):
        """

        :param x:
        :param n_blocks: 5  , dtype = int
        :param block_out_ch: 32 , dtype = int
        :param block_stride: 2 , dtype = int
        :return:
        """
        layer=x
        for idx in range(n_blocks):
            if idx == n_blocks-1:
                block_stride = block_stride
            else:
                block_stride = 1
            layer = self._block(layer , block_out_ch=block_out_ch , block_stride = block_stride , block_n=idx)
        return layer
    def _block(self , x , block_out_ch  , block_stride  , block_n):

        shortcut = x
        layer=x
        m=4 if self.use_bottlenect else 1
        out_ch = m * block_out_ch
        """ bottlenect layer """
        if self.use_bottlenect:
            with tf.variable_scope('bottlenect_{}'.format(block_n)):
                layer = batch_norm_layer(layer , self.train_phase  , 'bn_0')
                layer = convolution2d('conv_0' , layer , out_ch = block_out_ch , k =1 , s =1 ) #fixed padding padding = "SAME"
                layer = batch_norm_layer(layer, self.train_phase, 'bn_1')
                layer = convolution2d('conv_1', layer, out_ch=block_out_ch, k=3,
                                      s=block_stride)  # fixed padding padding = "SAME"
                layer = batch_norm_layer(layer, self.train_phase, 'bn_2')
                layer = convolution2d('conv_2', layer, out_ch=block_out_ch, k=1, s=1)  # fixed padding padding = "SAME"

        else: #""" redisual layer """
            with tf.variable_scope('residual_{}.'.format(block_n)):
                layer = convolution2d('conv_0' , layer , block_out_ch , k=3 , s=block_stride) # in here , if not block_stride = 1 , decrease image size
                layer = batch_norm_layer(layer , self.train_phase,'bn_0' )
                layer = convolution2d('conv_1', layer, block_out_ch, k=3, s=1)

        if not block_stride ==1: # image size 가 줄어들면 shortcut layer 의 이미지도 줄여야 한다
            shortcut_layer = convolution2d('shortcut_layer' , out_ch = out_ch , k =1 , s= block_stride)

        return shortcut + layer

    def _logit(self ,x  , phase_train):
        if self.logit_type == 'gap':
            logit=gap('gap' , x , out_ch = self.n_classes)
        elif self.logit_type == 'fc':

            layer=tf.cond(phase_train , lambda: tf.nn.dropout(x , keep_prob=0.5) , lambda: layer)
            logit=affine('fc' , layer ,out_ch=self.n_classes , keep_prob=  )
        else :
            print 'Not Implemneted , Sorry '

        return logit
