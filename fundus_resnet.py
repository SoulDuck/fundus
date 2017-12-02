import data
import resnet
import tensorflow as tf
import numpy as np
import aug
import cnn
import utils

# tensorboard
# model save
# run training using global step
train_imgs, train_labs, train_filenames, test_imgs, test_labs, test_filenames=data.type2('./fundus_300')
train_imgs=train_imgs/255.
n_classes = 10

x_ = tf.placeholder(dtype = tf.float32 , shape=[None ,299 ,299 ,3 ])
y_ = tf.placeholder(dtype = tf.float32 , shape=[None , n_classes] )
phase_train = tf.placeholder(dtype = tf.bool , name = 'phase_train')
aug_x_=aug.aug_tensor_images(x_ , phase_train ,  img_size_cropped=224 )
n_filters_per_box = [8, 16, 32, 64]
n_blocks_per_box = [2, 2, 2, 2]
stride_per_box = [1, 2, 2, 2]
use_bottlenect = False

model = resnet.Resnet(aug_x_, phase_train, n_filters_per_box, n_blocks_per_box, stride_per_box, \
                       use_bottlenect, n_classes=10, activation=tf.nn.relu, logit_type='gap')
logit=model.logit
pred,pred_cls , cost , train_op,correct_pred ,accuracy=cnn.algorithm( logit , y_ , learning_rate=0.01 , optimizer='AdamOptimizer')


"""----------------------------------------------------------------------------------------------------------------
                                                Make Session                                 
----------------------------------------------------------------------------------------------------------------"""
config = tf.ConfigProto()
config.gpu_options.allow_growth= True
saver = tf.train.Saver(max_to_keep=10000000)
sess=tf.Session(config=config)
init = tf.group( tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)
logs_path='./logs/fundus_resnet'
tb_writer =tf.summary.FileWriter(logs_path)
tb_writer.add_graph(tf.get_default_graph())
best_acc_root = './model/fundus_resnet_type2/best_acc'
best_loss_root = './model/fundus_resnet_type2/best_loss'



test_imgs_list, test_labs_list = utils.divide_images_labels_from_batch(test_imgs, test_labs, batch_size=60)
test_imgs_labs = zip(test_imgs_list, test_labs_list)

max_acc , min_loss = 0, 100000
for step in range(60000):
    batch_xs, batch_ys = data.next_batch(train_imgs, train_labs, batch_size=60)
    _ , loss, acc = sess.run(fetches=[train_op , cost ,accuracy ] , feed_dict= {x_ : batch_xs, y_ : batch_ys, phase_train : True })

    if step % 100 == 0:
        # Get Validation Accuracy and Loss
        pred_list, cost_list = [], []
        for batch_xs , batch_ys in test_imgs_labs:
            batch_pred , batch_cost = sess.run(fetches=[pred ,cost ], feed_dict={x_: batch_xs, y_: batch_ys, phase_train: False})
            pred_list.extend(batch_pred)
            cost_list.append(batch_cost)
        val_acc = utils.get_acc(pred_list , test_labs)
        val_cost =  np.sum(cost_list)/float(len(cost_list))
        utils.save_model(sess, saver, max_acc, min_loss, val_acc, val_cost, best_acc_root, best_loss_root,
                         step)
        utils.write_acc_loss(tb_writer , prefix='test' , loss =val_acc , acc =val_cost )


        print 'train acc :{:06.4f} train loss : {:06.4f} val acc : {:06.4f} val loss : {:06.4f}'.format(acc , loss,val_acc , val_cost)







