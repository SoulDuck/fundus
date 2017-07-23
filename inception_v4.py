
from cnn import convolution2d , batch_norm_layer , affine , max_pool , convolution2d_manual , avg_pool
import tensorflow as tf
#ef convolution2d(name,x,out_ch,k=3 , s=2 , padding='SAME'):
def stem(name , x):
    with tf.variable_scope(name) as scope:
        layer=convolution2d('cnn_0',x,32,k=3,s=2 , padding='VALID')
        layer = convolution2d('cnn_1',layer, 32, k = 3, s = 1, padding = 'VALID')
        layer = convolution2d('cnn_2', layer, 64, k=3, s=1, padding='SAME')
        layer_1 = max_pool('max_3', layer, k=3, s=2, padding='VALID')
        layer_2 = convolution2d('cnn_3_1', layer, 96, k=3, s=2, padding='VALID')
        layer_join=tf.concat([layer_1,layer_2] , axis=3 , name='join')
        print 'layer_name :','join'
        print 'layer_shape :',layer_join.get_shape()
    return layer_join
def stem_1(name , x ):
    with tf.variable_scope(name) as scope:
        layer = convolution2d('cnn_0', x, 64, k=1, s=1)
        layer = convolution2d('cnn_1', layer, 96, k=3, s=1, padding='VALID')
        layer_ = convolution2d('cnn__0', x, 64, k=1, s=1)
        layer_ = convolution2d_manual('cnn__1', layer_, 64, k_h=7,k_w=1, s=1)
        layer_ = convolution2d_manual('cnn__2', layer_, 64, k_h=1,k_w=7,s=1 )
        layer_ = convolution2d('cnn__3', layer_, 96, k=3, s=1, padding='VALID')

        layer_join = tf.concat([layer, layer_], axis=3, name='join')
        print 'layer_name :','join'
        print 'layer_shape :',layer_join.get_shape()
    return layer_join
def stem_2(name ,x ):
    with tf.variable_scope(name) as scope:
        layer= convolution2d('cnn_0' , x, 192,k=3,s=2,padding='VALID')
        layer_=max_pool('max__0' , x, k=3 , s=2 , padding = 'VALID')
        layer_join = tf.concat([layer , layer_] , axis = 3 ,name='join')
        print 'layer_name :','join'
        print 'layer_shape :',layer_join.get_shape()
    return layer_join



def reductionA(name,x ):

    with tf.variable_scope(name) as scope:
        layer_ =max_pool('max_pool_0' ,x, k=3, s=2 ,padding='VALID')

        layer__ =convolution2d('cnn__0' ,x,192 , k=3 , s=2 , padding='VALID'  )

        layer___ = convolution2d('cnn___0',x,224, k=1, s=1, padding='SAME')
        layer___ = convolution2d('cnn___1',layer___,256, k=3, s=1, padding='SAME')
        layer___ = convolution2d('cnn___2',layer___,385, k=3, s=2, padding='VALID')

        layer_join=tf.concat([layer_ , layer__ , layer___ ], axis=3 , name='join')
        print 'layer_name :','join'
        print 'layer_shape :',layer_join.get_shape()
    return layer_join

def reductionB(name , x):
    with tf.variable_scope(name) as scope:
        layer_ = max_pool('max_pool_0',x, k=3, s=2, padding='VALID')

        layer__ = convolution2d('cnn__0',x,192, k=1, s=1, padding='SAME')
        layer__ = convolution2d('cnn__1' ,layer__, 192,k=3 ,s=2 ,padding='VALID')

        layer___ = convolution2d('cnn___0',x,256, k=1, s=1, padding='SAME')
        layer___ = convolution2d_manual('cnn___1',layer___,256, k_h=1 , k_w=7, s=1, padding='SAME')
        layer___ = convolution2d_manual('cnn___2',layer___,320, k_h=7, k_w=1, s=1, padding='SAME')
        layer___ = convolution2d('cnn___3',layer___, 320,k=3, s=2, padding='VALID')

        layer_join=tf.concat([layer_ , layer__ , layer___] , axis=3 , name='join')
        print 'layer_name :','join'
        print 'layer_shape :',layer_join.get_shape()
    return layer_join

def reductionC(name ,x ):
    with tf.variable_scope(name) as scope:
        layer=max_pool('max_pool0' , x ,3,2 , padding='VALID')

        layer_=convolution2d('cnn_0',x , 256,1,1)
        layer_=convolution2d('cnn_1',layer_, 384 , 3,1 , padding='VALID')

        layer__ = convolution2d('cnn_0', x, 256, 1, 1)
        layer__ = convolution2d('cnn_1', layer__, 256, 3, 1 , padding='VALID')

        layer___= convolution2d('cnn__0',x, 256 , 1,1)
        layer___= convolution2d('cnn__1',layer___,256 , 3,1)
        layer___= convolution2d('cnn__2',layer___,256 , 3,1 , padding='VALID')

        layer_join= tf.concat([layer , layer_ , layer__ , layer___], axis=3 , name='join' )
        return layer_join



def blockB(name , x):
    with tf.variable_scope(name) as scope:
        layer=avg_pool('avg_pool', x, k=2 , s=1)
        layer=convolution2d('cnn',layer,128,k=1,s=1)

        layer_=convolution2d('cnn_0',x,384,k=1,s=1)

        layer__=convolution2d('cnn__0',x,192,k=1,s=1)
        layer__ = convolution2d_manual('cnn__1', layer__, 224, k_h=1 , k_w=7, s=1 )
        layer__ = convolution2d_manual('cnn__2', layer__, 256, k_h=1 , k_w=7, s=1 )

        layer___=convolution2d('cnn___0',x,192,k=1,s=1)
        layer___=convolution2d_manual('cnn___1',layer___,192,k_h=1,k_w=7,s=1)
        layer___=convolution2d_manual('cnn___2',layer___,224,k_h=7,k_w=1,s=1)
        layer___=convolution2d_manual('cnn___3',layer___,224,k_h=1,k_w=7,s=1)
        layer___=convolution2d_manual('cnn___4',layer___,256,k_h=7,k_w=1,s=1)

        layer_join=tf.concat([layer, layer_ , layer__ , layer___] , axis=3 , name ='join')
        print 'layer_name :','join'
        print 'layer_shape :',layer_join.get_shape()
    return layer_join

def blockA(name , x):
    with tf.variable_scope(name) as scope:
        layer = avg_pool('avg_pool', x, k=2, s=1)
        layer = convolution2d('cnn', layer, 96, k=1, s=1)

        layer_ = convolution2d('cnn_0', x, 96, k=1, s=1)

        layer__ = convolution2d('cnn__0', x, 64, k=1, s=1)
        layer__ = convolution2d('cnn__1', layer__,96, k=3, s=1)

        layer___ = convolution2d('cnn___0', x,64, k=1, s=1)
        layer___ = convolution2d('cnn___1',layer___,96, k=3, s=1)
        layer___ = convolution2d('cnn___2',layer___,96, k=3, s=1)

        layer_join = tf.concat([layer, layer_, layer__, layer___], axis=3, name='join')
        print 'layer_name :', 'join'
        print 'layer_shape :', layer_join.get_shape()
    return layer_join

def blockC(name , x):
    with tf.variable_scope(name) as scope:
        layer = avg_pool('avg_pool', x, k=2, s=1)
        layer = convolution2d('cnn', layer, 256, k=1, s=1)

        layer_ = convolution2d('cnn_0', x, 256, k=1, s=1)

        layer__ = convolution2d('cnn__0',x, 384, k=1, s=1)
        layer__0 = convolution2d_manual('cnn__1_0',layer__ , 256, k_h=1,k_w=3, s=1)
        layer__1 = convolution2d_manual('cnn__1_1',layer__ , 256, k_h=3,k_w=1, s=1)

        layer___ = convolution2d('cnn___0', x,384 ,k=1, s=1)
        layer___ = convolution2d_manual('cnn___1', layer___,448, k_h=1, k_w=3 ,s=1)
        layer___ = convolution2d_manual('cnn___2', layer___,512, k_h=3 , k_w=1, s=1)
        layer___0 = convolution2d_manual('cnn___3_0', layer___, 256, k_h=3,k_w=1, s=1)
        layer___1 = convolution2d_manual('cnn___3_1', layer___,256, k_h=1,k_w=3, s=1)
        layer_join = tf.concat([layer, layer_, layer__0, layer__1 ,layer___0 , layer___1], axis=3, name='join')
        print 'layer_name :', 'join'
        print 'layer_shape :', layer_join.get_shape()
        return layer_join

def resnet_blockA(name ,x):
    with tf.variable_scope(name) as scope:
        layer=convolution2d('cnn0',x,128,1,1)
        layer_ = convolution2d('cnn_0', x, 128, 1, 1)
        layer_ = convolution2d_manual('cnn_1', layer_, 128, k_h=1,k_w=7 ,s=1)
        layer_ = convolution2d_manual('cnn_2', layer_, 128, k_h=7,k_w=1 ,s=1)
        layer_join=tf.concat([layer , layer_], axis=3 , name='join')
        layer_join=convolution2d('layer_join_cnn' , layer_join , 897 , 1,1)
        print 'layer_name :', 'join'
        print 'layer_shape :', layer_join.get_shape()

        if x.get_shape()[-1] != layer_join.get_shape()[-1]:
            x=convolution2d('upscale_dimension',x, layer_join.get_shape()[-1] , k=1,s=1)
        layer_join=tf.add(x,layer_join , 'add')
        return layer_join
def resnet_blockB(name , x):
    with tf.variable_scope(name) as scope:

        layer = convolution2d('cnn0', x, 32, 1, 1)
        layer_ = convolution2d('cnn_0', x, 32, 1, 1)
        layer_ = convolution2d('cnn_1', layer_, 32, 3, 1)
        layer__ = convolution2d('cnn__0', x, 32, 1, 1)
        layer__ = convolution2d('cnn__1', layer__, 32, 3, 1)
        layer__ = convolution2d('cnn__2', layer__, 32, 3, 1)

        layer_join=tf.concat([layer , layer_ , layer__] , axis=3 , name='join')
        layer_join=convolution2d('layer_join_cnn' , layer_join ,256, 1,1 )

        if x.get_shape()[-1] != layer_join.get_shape()[-1]:
            x=convolution2d('upscale_dimension',x, layer_join.get_shape()[-1] , k=1,s=1)
        layer_join=tf.add(x,layer_join )
        return layer_join

def resnet_blockC(name, x):
    with tf.variable_scope(name) as scope:
        layer = convolution2d('cnn0', x, 192, 1, 1)
        layer_ = convolution2d('cnn_0', x, 192, 1, 1)
        layer_ = convolution2d_manual('cnn_1', layer_, 192, k_h=1, k_w=3, s=1)
        layer_ = convolution2d_manual('cnn_2', layer_, 192, k_h=3, k_w=1, s=1)
        layer_join = tf.concat([layer, layer_], axis=3, name='join')
        layer_join = convolution2d('layer_join_cnn', layer_join, 1792, 1, 1)
        if x.get_shape()[-1] != layer_join.get_shape()[-1]:
            x=convolution2d('upscale_dimension',x, layer_join.get_shape()[-1] , k=1,s=1)
        layer_join = tf.add(x, layer_join, 'add')
        print 'layer_name :', 'join'
        print 'layer_shape :', layer_join.get_shape()
        return layer_join
def structure_A(x_):
    print 'stem A -> B -> C -> blockA -> reductionA -> blockB -> reduction B -> blockC'

    layer = stem('stem', x_)
    layer = stem_1('stem_1', layer)
    layer = stem_2('stem_2', layer)
    layer = blockA('blockA_0', layer)
    layer = reductionA('reductionA', layer)
    layer = blockB('blockB_0', layer)
    layer = reductionB('reductionB', layer)
    layer = blockC('blockC_0', layer)
    top_conv = tf.identity(layer, name='top_conv')
    return top_conv

def structure_B( x_ , phase_train):
    print 'stem A -> B -> C -> blockA -> reductionA -> blockB -> reduction B -> blockC'
    layer = stem('stem', x_)
    layer=batch_norm_layer(layer,phase_train,'stem_bn')
    layer = stem_1('stem_1', layer)
    layer=batch_norm_layer(layer,phase_train,'stem1_bn')
    layer = stem_2('stem_2', layer)
    layer=batch_norm_layer(layer,phase_train,'stem2_bn')
    layer = blockA('blockA_0', layer)
    layer = reductionA('reductionA', layer)
    layer = blockB('blockB_0', layer)
    layer = batch_norm_layer(layer,phase_train,'reductionA_bn')
    layer = reductionB('reductionB', layer)
    layer = blockC('blockC_0', layer)
    layer = tf.identity(layer, name='top_conv')
    return layer