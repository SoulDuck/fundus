from cnn import convolution2d , affine , dropout , algorithm
import tensorflow as tf
import numpy as np
from fundus_processing import dense_crop
import os
from data import next_batch,get_train_test_images_labels
class network(object):
    def __init__(self , conv_filters , conv_strides , conv_out_channels , fc_out_channels , n_classes , data_dir='./'):

        self.conv_filters = conv_filters
        self.conv_strides = conv_strides
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.n_classes = n_classes
        self.data_dir = data_dir
        self.next_batch = next_batch
        self.get_train_test_images_labels  = get_train_test_images_labels
        # building network
        self._input()
        self._build()
        self._start_session()


    def _input(self):
        fg_imgs = np.load(os.path.join(self.data_dir, 'fg_images.npy'))
        bg_imgs = np.load(os.path.join(self.data_dir, 'bg_images.npy'))
        n_fg, h, w, ch = np.shape(fg_imgs)
        self.train_imgs , self.train_labs , self.val_imgs ,self.val_labs=self.get_train_test_images_labels(fg_imgs , bg_imgs[:n_fg])

        print 'train_imgs',len(self.train_labs)
        print 'val_imgs', len(self.tval_labs)

        self.x_ = tf.placeholder(dtype=tf.float32, shape=[None, h , w, ch], name='x_')
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_classes], name='y_')
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        self.phase_train = tf.placeholder(dtype=tf.bool)
        self.lr = tf.placeholder(dtype = tf.float32)

    def _build(self):
        layer=self.x_
        for i in range(len(self.conv_filters)):
            k=self.conv_filters[i]
            s=self.conv_strides[i]
            out_ch = self.conv_out_channels[i]
            layer = convolution2d('conv_{}'.format(i), x=layer , out_ch=out_ch, k=k, s=s) # activation = relu # dropout X
        self.top_conv = tf.identity(layer , name = 'top_conv')


        # Building fully connected layer...
        layer=self.top_conv
        for i in range(len(self.fc_out_channels)):
            out_ch= self.fc_out_channels[i]
            layer=affine(name = 'fc_{}'.format(i) ,x= layer , out_ch=out_ch )
            layer=dropout(layer , phase_train=self.phase_train , keep_prob=0.5)

        #make Logits
        self.logits=affine(name='logits' , x=layer , out_ch=self.n_classes)


    def _algorithm(self):
        self.pred, self.pred_cls, self.cost, self.train_op, self.correct_pred, self.accuracy = algorithm(
            y_conv=self.logits, y_=self.y_,
            learning_rate=self.lr, optimizer='sgd')
    def _start_session(self):
        self.sess = tf.Session()
        init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
        self.sess(init)

    def train(self , max_iter):
        for i in range(max_iter):
            batch_xs , batch_ys=self.next_batch(self.train_imgs , self.train_labs)
            next_batch()
            feed_dict={self.x_ : batch_xs , self.y_: batch_ys ,self.phase_train: True , self.lr:0.01}
            self.sess.run(self.train_op , feed_dict= feed_dict )




if __name__=='__main__':
    conv_filters=[3,3,3,3,3]
    conv_strides=[2,2,1,1,2,]
    conv_out_channels=[64,64,128,128,256]
    fc_out_channels=[1024,1024]
    n_classes=2
    network=network(conv_filters , conv_strides , conv_out_channels , fc_out_channels , n_classes)







