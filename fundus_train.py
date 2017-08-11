# -*- coding: utf-8 -*-
import tensorflow as tf
from cnn import convolution2d, max_pool, algorithm, affine, batch_norm_layer, gap
import inception_v4
import data
import numpy as np
import utils
from inception_v4 import stem, stem_1, stem_2, reductionA, reductionB, blockA, blockB, blockC
import cam
import aug
import random
import argparse
import os


def train(max_iter ,  batch_size ,learning_rate , nx=[30, 30, 30, 5, 5, 5, 70], structure='inception_A', restored_model_folder_path=None , restored_path_folder_path=None):
    ##########################setting############################
    image_height, image_width, image_color_ch, n_classes, \
    train_list_imgs_labs, test_list_imgs_labs, train_list_file_paths, test_list_file_paths,names = data.fundus_300x300(reload_folder_path=restored_path_folder_path)

    if restored_model_folder_path is None:
        restored_model_folder_path = utils.make_folder('./cnn_model/', 'fundus/')
    if restored_path_folder_path is None:
        restored_path_folder_path= utils.make_folder('./paths/' , 'fundus/')
    graph_saved_folder_path = utils.make_folder('./graph/', 'fundus/')
    log_saved_folder_path = utils.make_folder('./log/', 'fundus/')
    log_saved_file_path = log_saved_folder_path + 'log.txt'
    f = open(log_saved_file_path , 'w+' , 0)

    test_save_folder_paths=map(lambda name :os.path.join(restored_path_folder_path  , name+'_test_paths.txt'),names )
    train_save_folder_paths = map(lambda name: os.path.join(restored_path_folder_path, name + '_train_paths.txt'),names)
    map(data.save_paths ,test_list_file_paths ,test_save_folder_paths )
    map(data.save_paths, train_list_file_paths, train_save_folder_paths)
    #save paths


    x_ = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, image_color_ch], name='x_')
    y_ = tf.placeholder(dtype=tf.int32, shape=[None, n_classes], name='y_')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
    ##########################structure##########################
    if structure == 'inception_A':
        top_conv=inception_v4.structure_A(x_)
    elif structure == 'inception_B':
        top_conv = inception_v4.structure_B(x_ , phase_train)

    y_conv = gap('gap', top_conv, 2)
    cam_ = cam.get_class_map('gap', top_conv, 0, image_height)
    #################fully connected#############################
    """
    layer=tf.contrib.layers.flatten(layer)
    print layer.get_shape()
    layer = affine('fully_connect', layer, 1024 ,keep_prob=0.5)
    y_conv=affine('end_layer' , layer , n_classes , keep_prob=1.0)
    """
    #############################################################
    # cam = get_class_map('gap', top_conv, 0, im_width=image_width)
    pred, pred_cls, cost, train_op, correct_pred, accuracy = algorithm(y_conv, y_, learning_rate)
    saver = tf.train.Saver()
    config = tf.ConfigProto(
        device_count={'GPU': 1},
        log_device_placement=True
    )
    sess = tf.Session(config=config)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    try:
        saver.restore(sess, restored_model_folder_path + 'best_acc.ckpt')
        print restored_model_folder_path+'model was restored!'
    except tf.errors.NotFoundError:
        print 'there was no model , make new model'
    ########################training##############################

    max_val = 0
    check_point = 100
    train_acc = 0;
    train_loss = 0;
    val_imgs, val_labs = data.make_batch(test_list_imgs_labs, nx, names=names)
    for step in range(max_iter):
        utils.show_progress(step, max_iter)
        if step % check_point == 0:
            # cam.inspect_cam(sess, cam_ , top_conv,test_imgs, test_labs, step , 50 , x_,y_ , y_conv  )
            imgs, labs = utils.divide_images_labels_from_batch(val_imgs, val_labs, batch_size)
            list_imgs_labs = zip(imgs, labs)
            val_acc_mean = [];
            val_loss_mean = []
            for val_imgs_, val_labs_ in list_imgs_labs:
                val_acc, val_loss = sess.run([accuracy, cost],
                                             feed_dict={x_: val_imgs_, y_: val_labs_, phase_train: False})
                val_acc_mean.append(val_acc);
                val_loss_mean.append(val_loss);
            val_acc = np.mean(np.asarray(val_acc_mean));
            val_loss = np.mean(np.asarray(val_loss_mean))
            utils.write_acc_loss(f, train_acc, train_loss, val_acc, val_loss)
            print '\n', val_acc, val_loss
            if val_acc > max_val:
                saver.save(sess, restored_model_folder_path + '/best_acc.ckpt')
                print 'model was saved!'
                max_val = val_acc
        # names = ['cataract', 'glaucoma', 'retina', 'retina_glaucoma','retina_cataract', 'cataract_glaucoma', 'normal']
        batch_xs, batch_ys = data.make_batch(test_list_imgs_labs,nx=[10,10,10,5,5,5,35] , names=names)
        batch_xs = aug.aug_level_1(batch_xs)
        train_acc, train_loss, _ = sess.run([accuracy, cost, train_op],
                                            feed_dict={x_: batch_xs, y_: batch_ys, phase_train: True})
        f.flush()
    f.close()
    utils.draw_grpah(log_saved_file_path , graph_saved_folder_path , check_point)


def train_with_specified_gpu(max_iter , batch_size , learning_rate,restored_model_folder_path=None , gpu_device='/gpu:0'):
    with tf.device(gpu_device):
        train(max_iter , batch_size, learning_rate)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", help='iteration')
    parser.add_argument("--batch_size" ,help='batch size ')
    parser.add_argument("--learning_rate" , help='learning rate ')
    parser.add_argument("--structure" , help = 'what structrue you need')
    parser.add_argument("--gpu",help='used gpu')

    args = parser.parse_args()
    #args.iter=int(args.iter)
    #args.batch_size = int(args.batch_size)
    #args.learning_rate =float(args.learning_rate)
    """
    debugging
    args.iter=100
    args.batch_size=10
    args.learning_rate=0.001
    args.structure='inception_A'
    """
    #train_with_redfree(args.iter , args.batch_size , args.learning_rate , args.structure , restored_model_folder_path=None)
    #train_with_specified_gpu(gpu_device='/gpu:1')
    train(max_iter=200 , batch_size=60 ,learning_rate = 0.01)