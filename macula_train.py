import tensorflow as tf
from cnn  import convolution2d , max_pool , algorithm , affine , batch_norm_layer , gap
import inception_v4
import data
import numpy as np
import utils
from inception_v4 import stem , stem_1 , stem_2 ,reductionA , reductionB , blockA , blockB , blockC
import cam
import aug
##########################setting############################
image_height, image_width, image_color_ch, n_classes, train_imgs_labs, test_imgs, test_labs = data.macula_299x299()

f=utils.make_log_txt() # make log and log folder
model_saved_folder_path=utils.make_folder('./cnn_model', 'macula')

x_ = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, image_color_ch], name='x_')
y_ = tf.placeholder(dtype=tf.int32, shape=[None, n_classes], name='y_')
phase_train=tf.placeholder(dtype=tf.bool , name='phase_train')
batch_size=60
##########################structure##########################

layer=stem('stem' , x_)
#batch_norm_layer(layer,phase_train,'stem_bn')
layer=stem_1('stem_1' , layer)
#batch_norm_layer(layer,phase_train,'stem1_bn')
layer=stem_2('stem_2' , layer)
#batch_norm_layer(layer,phase_train,'stem2_bn')
layer=blockA('blockA_0' , layer)
layer=reductionA('reductionA' , layer)
layer=blockB('blockB_0' , layer)
#batch_norm_layer(layer,phase_train,'reductionA_bn')
layer=reductionB('reductionB' , layer)
layer=blockC('blockC_0' , layer)
top_conv=tf.identity(layer ,name='top_conv' )
#batch_norm_layer(layer,phase_train,'reductionB_bn')
y_conv=gap('gap' ,top_conv ,2 )
cam_=cam.get_class_map('gap',top_conv , 0 ,image_height )

#################fully connected#############################
"""
layer=tf.contrib.layers.flatten(layer)
print layer.get_shape()
layer = affine('fully_connect', layer, 1024 ,keep_prob=0.5)
y_conv=affine('end_layer' , layer , n_classes , keep_prob=1.0)
"""
#############################################################
#cam = get_class_map('gap', top_conv, 0, im_width=image_width)
pred, pred_cls, cost, train_op, correct_pred, accuracy = algorithm(y_conv, y_, 0.001)
saver = tf.train.Saver()
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
try:
    saver.restore(sess, model_saved_folder_path+'/best_acc.ckpt')
    print 'model was restored!'
except tf.errors.NotFoundError:
    print 'there was no model'
########################training##############################
max_val = 0
max_iter=500000
check_point = 50
train_acc=0;train_loss=0;

for step in range(max_iter):
    utils.show_progress(step,max_iter)
    if step % check_point == 0:
        cam.inspect_cam(sess, cam_ , top_conv,test_imgs, test_labs, step , 50 , x_,y_ , y_conv  )
        imgs,labs=utils.divide_images_labels_from_batch(test_imgs,test_labs,batch_size)
        list_imgs_labs=zip(imgs,labs)
        val_acc_mean=[];val_loss_mean=[]
        for img,lab in list_imgs_labs:
            val_acc, val_loss = sess.run([accuracy, cost],feed_dict={x_: img, y_: lab, phase_train: False})
            val_acc_mean.append(val_acc);
            val_loss_mean.append(val_loss)
        val_acc=np.mean(np.asarray(val_acc_mean));val_loss=np.mean(np.asarray(val_loss_mean))
        utils.write_acc_loss(f,train_acc,train_loss ,val_acc , val_loss)
        print '\n',val_acc, val_loss
        if val_acc > max_val:
            saver.save(sess, model_saved_folder_path+'/best_acc.ckpt')
            print 'model was saved!'
            max_val=val_acc
    batch_xs, batch_ys = data.make_train_batch(train_imgs_labs[0], train_imgs_labs[1], train_imgs_labs[2],train_imgs_labs[3])
    batch_xs=aug.aug_level_1(batch_xs)
    train_acc, train_loss, _ = sess.run([accuracy, cost, train_op], feed_dict={x_: batch_xs, y_: batch_ys , phase_train:True})

