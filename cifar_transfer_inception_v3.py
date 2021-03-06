# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle
import os
import utils
import data
import glob
from cnn import affine, algorithm, lr_schedule
from PIL import Image
from sklearn.decomposition import PCA
import PIL
import transfer
import argparse
from cifar_  import input


parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', type=int , default=100000);
parser.add_argument('--optimizer', type=str , default='adam');
args = parser.parse_args()
"""
 by re-routing the output of the original model just prior to its classification layers
 and instead use a new classifier that we had created.
 Because the original model was 'frozen' its weights could not be further optimized, 

"""

"""----------------------------------------------------------------------------------------------------------------
                                                Input Data
----------------------------------------------------------------------------------------------------------------"""

inception_v3_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
transfer.download_and_extract_model(url=inception_v3_url, data_dir='./pretrained_models/inception_v3')
ckpt_dir = 'inception_v3_pretrained_cifar'
train_imgs  , train_labs , test_imgs ,test_labs =input.get_cifar_images_labels(True , './cifar_/cifar_10/cifar-10-batches-py')
n_classes=10

x_ = tf.placeholder(dtype=tf.float32, shape=[None, 2048])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y_')
phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
lr_ = tf.placeholder(dtype=tf.float32, name='learning_rate')

"""----------------------------------------------------------------------------------------------------------------
                                                define Model 
----------------------------------------------------------------------------------------------------------------"""

model = transfer.Transfer_inception_v3('./pretrained_models/inception_v3', x_, phase_train, 0.5, [n_classes])
train_imgs = model.images2caches('./pretrained_models/inception_v3/cifar10_train_cache.pkl', train_imgs)
test_imgs = model.images2caches('./pretrained_models/inception_v3/cifar10_test_cache.pkl', test_imgs)
train_imgs = train_imgs / 255.
test_imgs = test_imgs / 255.
pred, pred_cls, cost, train_op, correct_pred, accuracy = algorithm(model.logits, y_=y_, learning_rate=lr_,
                                                                   optimizer=args.optimizer, use_l2_loss=False)

"""----------------------------------------------------------------------------------------------------------------
                                                Make Session                                 
----------------------------------------------------------------------------------------------------------------"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver(max_to_keep=10000000)
last_model_saver = tf.train.Saver(max_to_keep=1)
sess = tf.Session(config=config)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)
logs_path = os.path.join('./logs', ckpt_dir)
tb_writer = tf.summary.FileWriter(logs_path)
tb_writer.add_graph(tf.get_default_graph())
model_root_dir = os.path.join('./model', ckpt_dir)
last_model_ckpt_dir = os.path.join(model_root_dir ,'last')
last_model_ckpt_path = os.path.join(last_model_ckpt_dir, 'model')
try:
    os.makedirs(last_model_ckpt_dir)
except Exception as e:
    pass;
start_step = utils.restore_model(saver=last_model_saver, sess=sess, ckpt_dir=last_model_ckpt_dir)

"""----------------------------------------------------------------------------------------------------------------
                                                Training Model                                  
----------------------------------------------------------------------------------------------------------------"""
batch_size = 120
lr_iters = [5000, 10000]
lr_values = [0.01, 0.01]
max_acc, min_loss = 0, 10000000
max_iter = args.max_iter;
for step in range(start_step, max_iter):
    lr = lr_schedule(step, lr_iters, lr_values)
    batch_xs, batch_ys = data.next_batch(train_imgs, train_labs, batch_size=batch_size)
    _, loss, acc = sess.run(fetches=[train_op, cost, accuracy],
                            feed_dict={x_: batch_xs, y_: batch_ys, phase_train: True, lr_: lr})
    #last_model_saver.save(sess, save_path=last_model_ckpt_path, global_step=step)
    if step % 100 == 0:
        # Get Validation Accuracy and Loss
        val_pred, val_cost = sess.run(fetches=[pred, cost],
                                      feed_dict={x_: test_imgs, y_: test_labs, phase_train: False})
        val_acc = utils.get_acc(val_pred, test_labs)

        max_acc, min_loss = utils.save_model(sess, max_acc, min_loss, val_acc, val_cost, step, model_root_dir,
                                             last_model_saver, saver)
        utils.write_acc_loss(tb_writer, prefix='test', loss=val_cost, acc=val_acc, step=step)
        utils.write_acc_loss(tb_writer, prefix='train', loss=loss, acc=acc, step=step)
        lr_summary = tf.Summary(value=[tf.Summary.Value(tag='learning_rate', simple_value=float(lr))])
        tb_writer.add_summary(lr_summary, step)
        print 'train acc :{:06.4f} train loss : {:06.4f} val acc : {:06.4f} val loss : {:06.4f}'.format(acc, loss,
                                                                                                        val_acc,
                                                                                                        val_cost)