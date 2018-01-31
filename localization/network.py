from cnn import convolution2d , affine , dropout , algorithm , logits
import tensorflow as tf
import numpy as np
from fundus_processing import dense_crop
import os
from data import next_batch,get_train_test_images_labels , divide_images_labels_from_batch ,divide_images , cls2onehot
from utils import get_acc,show_progress ,plot_images , save_model , make_saver , restore_model
import mnist
import random
from aug import random_rotate_images
from PIL import Image
class network(object):
    def __init__(self, conv_filters, conv_strides, conv_out_channels, fc_out_channels, n_classes, batch_size,
                 data_dir='./', restore_type='last'):

        self.conv_filters = conv_filters
        self.conv_strides = conv_strides
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.restore_type = restore_type #'last' or 'best_acc'

        #bring method from the other method
        self.next_batch = next_batch
        self.get_train_test_images_labels  = get_train_test_images_labels
        self.divide_images_labels_from_batch = divide_images_labels_from_batch
        self.get_acc = get_acc
        self.make_saver = make_saver
        self.save_model = save_model
        self.restore_model = restore_model
        self.cls2onehot = cls2onehot
        # building network
        self._input()
        self._model()
        self._algorithm()
        self._start_session()

        #
        self.best_acc=0
        self.best_loss=10000000
        self.acc=0
        self.loss=10000000
    def _input(self):

        self.train_fg_imgs = np.load(os.path.join(self.data_dir, 'train_fg_images.npy'))/255.
        self.train_bg_imgs = np.load(os.path.join(self.data_dir, 'train_bg_images.npy'))/255.
        test_fg_imgs = np.load(os.path.join(self.data_dir, 'test_fg_images.npy'))/255.
        test_bg_imgs = np.load(os.path.join(self.data_dir, 'test_bg_images.npy'))/255.



        self.val_imgs = np.vstack((test_fg_imgs, test_bg_imgs))
        self.val_labs = np.vstack((self.cls2onehot(np.zeros([len(test_fg_imgs)]), 2),
                                  self.cls2onehot(np.ones([len(test_bg_imgs)]), 2)))
        #divide images into train , validation dataset
        # for mnist # if you want test toy sample , uncommnet below line
        # self.train_imgs = mnist.train_imgs;self.train_labs = mnist.train_labs;self.val_imgs =mnist.val_imgs;self.val_labs = mnist.val_labs
        # show n train  , n validation dataset
        self.n_fg, h, w, ch = np.shape(self.train_fg_imgs) #to set h , w ,ch
        self.n_bg, self.h, self.w, self.ch = np.shape(self.train_bg_imgs)  # to set h , w ,ch
        self.x_ = tf.placeholder(dtype=tf.float32, shape=[None, h , w, ch], name='x_')
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_classes], name='y_')
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        self.phase_train = tf.placeholder(dtype=tf.bool)
        self.lr = tf.placeholder(dtype = tf.float32)

    def _model(self):
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
        self.logits=logits(name='logits' , x=layer, n_classes=self.n_classes)

    def _algorithm(self):
        self.pred, self.pred_cls, self.cost, self.train_op, self.correct_pred, self.accuracy = algorithm(
            y_conv=self.logits, y_=self.y_,
            learning_rate=self.lr, optimizer='sgd',use_l2_loss=False)

    def _start_session(self):
        self.last_saver , self.best_saver=self.make_saver()
        self.sess = tf.Session()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)
        ckpt_dir = os.path.join('./model' , self.restore_type)
        self.global_step=self.restore_model(self.last_saver ,self.sess , ckpt_dir , self.restore_type)
    def _make_batch(self):
        fg_indices = random.sample(range(self.n_fg), int(self.batch_size / 2))
        bg_indices = random.sample(range(self.n_fg), int(self.batch_size - len(fg_indices)))
        fg_batch_xs = self.train_fg_imgs[fg_indices]
        bg_batch_xs = self.train_bg_imgs[bg_indices]
        fg_batch_ys = self.cls2onehot(np.zeros(len(fg_indices)), 2)
        bg_batch_ys = self.cls2onehot(np.ones(len(bg_indices)), 2)


        batch_xs = np.vstack((fg_batch_xs, bg_batch_xs))
        batch_ys = np.vstack((fg_batch_ys, bg_batch_ys))
        indices=random.sample(range(len(batch_ys)) , len(batch_ys))
        batch_xs=batch_xs[indices]
        batch_ys = batch_ys[indices]

        batch_xs=(batch_xs)
        batch_xs=random_rotate_images(batch_xs)


        return batch_xs , batch_ys
    def train(self , max_iter):
        for i in range(self.global_step,max_iter):
            batch_xs , batch_ys=self._make_batch()
            if np.max(batch_xs)>1:
                batch_xs=batch_xs/255.
            show_progress(i ,max_iter)
            feed_dict={self.x_ : batch_xs  , self.y_: batch_ys ,self.phase_train: True , self.lr:0.01}
            _,train_acc , train_loss =self.sess.run([self.train_op ,self.accuracy , self.cost], feed_dict= feed_dict )
            self.global_step+=1
        return train_acc , train_loss

    def val(self):
        all_pred=[]
        mean_cost=[]
        if np.max(self.val_imgs)>1:
            self.val_imgs=self.val_imgs/255.
        batch_imgs_list , batch_labs_list=self.divide_images_labels_from_batch(self.val_imgs ,self.val_labs , self.batch_size)
        for i in range(len(batch_labs_list)):
            batch_ys = batch_labs_list[i]
            batch_xs = batch_imgs_list[i]
            feed_dict = {self.x_: batch_xs, self.y_: batch_ys, self.phase_train: False, self.lr: 0.1}
            pred,cost=self.sess.run([self.pred ,self.cost], feed_dict=feed_dict)
            all_pred.extend(pred)
            mean_cost.append(cost)

        self.acc=self.get_acc(true=self.val_labs , pred=all_pred)
        self.loss=np.mean(mean_cost)
        self.best_acc, self.best_loss = self.save_model(self.sess, self.best_acc, self.best_loss, self.acc, self.loss,
                                                      self.global_step, './model', self.last_saver, self.best_saver)

        print self.acc, self.loss




class detection(network):
    def __init__(self , img_dir , crop_size):
        conv_filters = [3, 3, 3, 3, 3]
        conv_strides = [2, 2, 1, 1, 2, ]
        conv_out_channels = [64, 64, 128, 128, 256]
        fc_out_channels = [1024, 1024]
        n_classes = 2

        # restore or train classification model
        self.divide_images = divide_images
        self.model=network(conv_filters, conv_strides, conv_out_channels, fc_out_channels, n_classes, 60 , restore_type='acc')
        self.img_dir = img_dir
        self.test_paths=self._load_test_imgs()
        self.crop_size = crop_size
        self.img_path=self._load_test_imgs()
    def _load_test_imgs(self):
        f=open('test_path.txt','r')
        img_paths=[]
        for line in f.readlines():
            name=os.path.splitext(os.path.split(line.replace('\n',''))[1])[0]
            img_paths.append(os.path.join(self.img_dir, name+'.png'))
        return img_paths

    def detect_target(self ,image):
        cropped_imgs, coords = self.dense_crop(image, self.crop_size, self.crop_size ,interval=75 )
        imgs_list =self.divide_images(cropped_imgs ,self.model.batch_size) #from network
        all_pred=[]
        for i in range(len(imgs_list)):
            show_progress(i , len(imgs_list))
            #labs=labs_list[i]
            imgs=imgs_list[i]
            if np.max(imgs)>1:
                imgs=imgs/255.
            all_pred.extend(
                self.model.sess.run(self.model.pred, feed_dict={self.model.x_: imgs, self.model.phase_train: False}))

        return all_pred ,coords

    def dense_crop(self,image , crop_height , crop_width , interval):
        """
         _________________
        | ____       ___  |
        ||    |-->->|   | |
        ||____|     |___| |

                ...
          ____       ____
        ||    |-->->|    ||
        ||____|     |____||
        |_________________|
        :param image:
        :param crop_height:
        :param crop_width:
        :param lr_flip:
        :param ud_flip:
        :return:
        """
        coords=[]
        cropped_images=[]
        img_h,img_w,ch=np.shape(image)
        n_h_move = img_h - crop_height + 1
        n_w_move = img_w - crop_width + 1
        for h in range(0,n_h_move,interval):
            for w in range(0,n_w_move,interval):
                x1 = w
                y1 = h
                x2 = w + crop_width
                y2 = h + crop_height
                coords.append((x1,y1,x2,y2))
                cropped_images.append(image[h: h + crop_height, w: w + crop_width, :])
        assert len(cropped_images) == len(coords)
        return np.asarray(cropped_images) , coords



if __name__=='__main__':
    """
    img_dir='/Users/seongjungkim/data/detection/resize'
    crop_size=75
    detection_model=detection(img_dir,crop_size)
    img=np.asarray(Image.open(detection_model.img_path[0]))
    pred=detection_model.detect_target(img)
    """

    conv_filters=[3,3,3,3,3]
    conv_strides=[2,2,1,1,2,]
    conv_out_channels=[64,64,128,128,256]
    fc_out_channels=[1024,1024]

    ##mnist version ###
    n_classes = 2
    model= network(conv_filters, conv_strides, conv_out_channels, fc_out_channels, n_classes, 60,)
    model.train(13)
    model.val()
    #n_classes=2
    #network=network(conv_filters , conv_strides , conv_out_channels , fc_out_channels , n_classes,60)








