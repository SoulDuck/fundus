#-*- coding:utf-8 -*-
import tensorflow as tf
from cnn import convolution2d, batch_norm_layer, affine, max_pool, avg_pool , gap
class Alexnet(object):
    def __init__(self , x_ , phase_train , conv_n_filters  , conv_k_sizes , conv_strides , fc_nodes,n_classes , activation , \
                 norm ,logit_type , ):
        # default Alexnet
        # 5 convolution 3 fully connected Layer
        # Conv [ k=11 , s=4 , pooling same , ch 96 ] --> max pooling [ k=3 ,s=2 ] --> lrn
        # --> Conv [ k=5 , s=1 , pooling same ch 256]--> max pooling [ k=3 ,s=2 ] --> lrn
        # --> Conv [ k=3 , s=1 , pooling same ch 384]--> Conv [ k=3 , s=1 , pooling same ch 384]
        # -->Conv [ k=3 , s=1 , pooling same ch 256] -->FC 4096 --> FC 4096 ->FC 1000 neuons
        #BN 을 쓸지 LRN 을 쓸지 결정
        print '#############################################################'
        print '                           AlexNet'
        print '#############################################################'

        self.x_ = x_
        self.phase_train = phase_train
        self.conv_n_filters=conv_n_filters
        self.conv_k_sizes=conv_k_sizes
        self.conv_strides=conv_strides
        self.fc_nodes = fc_nodes
        self.n_classes = n_classes
        self.activation = activation
        self.norm=norm
        self.logit_type = logit_type
        self.n_conv_layers = 5
        self.n_fc_layers = 3  # fc_0 , fc_1 , fc_2
        self.logit=self._build_model()

    def _norm(self , x_ , phase_train , scope_bn):
        if self.norm == 'BN':
            layer=batch_norm_layer(x_  , phase_train , self.norm+'_'+scope_bn)
        elif self.norm =='LRN':
            layer=tf.nn.lrn(input=x_, name=self.norm+'_'+scope_bn)
        else :
            raise AssertionError
        return layer
    def _build_model(self):
        for i in range(self.n_conv_layers):
            if i < 2:
                layer=convolution2d(name='conv_{}'.format(i) , x=self.x_ , out_ch=self.conv_n_filters[i] \
                              , k=self.conv_k_sizes[i] ,s=self.conv_strides[i])
                layer=max_pool(name='max_pool_{}'.format(i) , x=layer , k=3 , s=2 )

                layer=self._norm(layer , self.phase_train , scope_bn='norm_{}'.format(i))
            else:
                layer = convolution2d(name='conv_{}'.format(i), x= layer , out_ch=self.conv_n_filters[i] \
                                      , k=self.conv_k_sizes[i], s=self.conv_strides[i])
            layer = tf.identity(layer , name = 'top_conv')
            layer = max_pool(name='max_pool_{}'.format(i), x=layer, k=3, s=2)

        if self.logit_type == 'fc':
            for j in range(self.n_fc_layers):
                layer=affine(name='fc_{}'.format(j) , x=layer , out_ch=self.fc_nodes[j])
        elif self.logit_type == 'gap':
            layer = gap('gap', layer , n_classes=self.n_classes)
        logit = tf.identity(layer, name='logits')
        return logit


if __name__ =='__main__':
    phase_train = tf.placeholder(dtype = tf.bool , name='phase_train')
    x_  = tf.placeholder(dtype = tf.float32  , shape = [ None , 32 ,32 , 3 ] , name = 'x_')
    conv_n_filters = [96, 256, 384, 384, 256]
    conv_k_sizes = [11, 5, 3, 3, 3]
    conv_strides = [2, 2, 1, 1, 1]
    fc_nodes=[4096 , 4096 , 2 ]
    model = Alexnet(x_ , phase_train , conv_n_filters , conv_k_sizes ,  conv_strides , fc_nodes ,\
                    n_classes=2 , activation='relu' , norm='BN' , logit_type='fc')










