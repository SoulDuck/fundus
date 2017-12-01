import sys , os
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import resnet
import aug
import cnn
import input
import data
import eval
import utils
import numpy as np

train_imgs ,train_labs , test_imgs ,test_labs=input.get_cifar_images_labels(onehot=True)
train_imgs=train_imgs/255.
n_classes = 10
x_ = tf.placeholder(dtype = tf.float32 , shape=[None ,32 ,32 ,3 ])
y_ = tf.placeholder(dtype = tf.float32 , shape=[None , n_classes] )
phase_train = tf.placeholder(dtype = tf.bool , name = 'phase_train')
aug_x_=aug.aug_tensor_images(x_ , phase_train ,  img_size_cropped=28 )
n_filters_per_box = [8, 16, 32, 64]
n_blocks_per_box = [2, 2, 2, 2]
stride_per_box = [1, 2, 2, 2]
use_bottlenect = False

model = resnet.Resnet(aug_x_, phase_train, n_filters_per_box, n_blocks_per_box, stride_per_box, \
                       use_bottlenect, n_classes=10, activation=tf.nn.relu, logit_type='gap')
logit=model.logit
pred,pred_cls , cost , train_op,correct_pred ,accuracy=cnn.algorithm( logit , y_ , learning_rate=0.01 , optimizer='AdamOptimizer')


### session start ###

sess=tf.Session()
init = tf.group( tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)

test_imgs_list, test_labs_list = utils.divide_images_labels_from_batch(test_imgs, test_labs, batch_size=60)
test_imgs_labs = zip(test_imgs_list, test_labs_list)
pred_list=[]
for i in range(60000):
    batch_xs, batch_ys = data.next_batch(train_imgs, train_labs, batch_size=60)
    _ , loss, acc = sess.run(fetches=[train_op , cost ,accuracy ] , feed_dict= {x_ : batch_xs, y_ : batch_ys, phase_train : True })
    if i % 100 == 0:
        print '{} {} '.format(loss, acc)
        for batch_xs , batch_ys in test_imgs_labs:
            pred_list.extend(sess.run(fetches=pred, feed_dict={x_: batch_xs, y_: batch_ys, phase_train: False}))
        print len(pred_list)
        exit()









