import tensorflow as tf
import numpy as np
import argparse
import alexnet
import data
import aug
import cnn
import os
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir' , default='alexnet')
parser.add_argument('--conv_n_filters' , nargs='+' , type=int , default= [96, 256, 384, 384, 256])
parser.add_argument('--conv_k_sizes' , nargs='+' , type=int , default=[11, 5, 3, 3, 3])
parser.add_argument('--conv_strides', nargs='+' , type=int , default=[2, 2, 1, 1, 1])
parser.add_argument('--fc_nodes' , nargs='+', type=int , default=[4096, 4096, 2])
parser.add_argument('--logit_type' , type=str  , choices=['gap', 'fc'] , default='gap')
parser.add_argument('--batch_size' , type=int , default= 60)
parser.add_argument('--activation')
parser.add_argument('--norm' , default='BN')
parser.add_argument('--color_aug' ,dest='use_color_aug' , action='store_true')
parser.add_argument('--no_color_aug', dest='use_color_aug', action='store_false')
parser.add_argument('--lr_iters' ,nargs='+', type=int, default=[2000 ,10000 , 40000 , 80000] )
parser.add_argument('--lr_values',nargs='+', type=float, default=[0.001 , 0.0007 , 0.0004 , 0.00001])
parser.add_argument('--l2_loss' , dest='use_l2_loss' , action='store_true')
parser.add_argument('--no_l2_loss' , dest='use_l2_loss' , action='store_false')
args=parser.parse_args()

"""----------------------------------------------------------------------------------------------------------------
                                                Input Data
----------------------------------------------------------------------------------------------------------------"""

train_imgs, train_labs, train_filenames, test_imgs, test_labs, test_filenames = data.type2('./fundus_300' , save_dir_name=args.ckpt_dir)
train_imgs = train_imgs / 255.
test_imgs = test_imgs / 255.
n_classes = 2




x_ = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
lr_ = tf.placeholder(dtype=tf.float32, name='learning_rate')
phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
aug_x_ = aug.aug_tensor_images(x_, phase_train, img_size_cropped=224 , color_aug=args.use_color_aug)



model = alexnet.Alexnet(x_ , phase_train , args.conv_n_filters , args.conv_k_sizes , \
                        args.conv_strides , args.fc_nodes , n_classes , args.activation , args.norm  , args.logit_type)

logit=model.logit
print np.shape(logit)
pred, pred_cls, cost, train_op, correct_pred, accuracy = cnn.algorithm(logit, y_, learning_rate=lr_,
                                                                       optimizer='AdamOptimizer')



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
logs_path = os.path.join('./logs', 'fundus_resnet', args.ckpt_dir)
tb_writer = tf.summary.FileWriter(logs_path)
tb_writer.add_graph(tf.get_default_graph())
best_acc_ckpt_dir = os.path.join('./model', args.ckpt_dir, 'best_acc')
best_loss_ckpt_dir = os.path.join('./model', args.ckpt_dir, 'best_loss')
last_model_ckpt_dir = os.path.join('./model', args.ckpt_dir, 'last_model')
last_model_ckpt_path = os.path.join(last_model_ckpt_dir, 'model')

try:
    os.makedirs(last_model_ckpt_dir)
except Exception as e:
    pass;
start_step = utils.restore_model(saver=last_model_saver, sess=sess, ckpt_dir=last_model_ckpt_dir)


"""----------------------------------------------------------------------------------------------------------------
                                                Training Model                                  
----------------------------------------------------------------------------------------------------------------"""

test_imgs_list, test_labs_list = utils.divide_images_labels_from_batch(test_imgs, test_labs, batch_size=60)
test_imgs_labs = zip(test_imgs_list, test_labs_list)

max_acc, min_loss = 0, 10000000
for step in range(start_step, 100000):
    lr = cnn.lr_schedule(step , args.lr_iters , args.lr_values)
    batch_xs, batch_ys = data.next_batch(train_imgs, train_labs, batch_size=args.batch_size)
    batch_xs=aug.random_rotate_image(batch_xs) # random rotate images

    _, loss, acc = sess.run(fetches=[train_op, cost, accuracy],
                            feed_dict={x_: batch_xs, y_: batch_ys, phase_train: True, lr_: lr})
    last_model_saver.save(sess, save_path=last_model_ckpt_path, global_step=step)
    if step % 100 == 0:
        # Get Validation Accuracy and Loss
        pred_list, cost_list = [], []
        for batch_xs, batch_ys in test_imgs_labs:
            batch_pred, batch_cost = sess.run(fetches=[pred, cost],
                                              feed_dict={x_: batch_xs, y_: batch_ys, phase_train: False})
            pred_list.extend(batch_pred)
            cost_list.append(batch_cost)
        val_acc = utils.get_acc(pred_list, test_labs)
        val_cost = np.sum(cost_list) / float(len(cost_list))
        max_acc, min_loss = utils.save_model(sess, saver, max_acc, min_loss, val_acc, val_cost, best_acc_ckpt_dir,
                                             best_loss_ckpt_dir,
                                             step)
        utils.write_acc_loss(tb_writer, prefix='test', loss=val_cost, acc=val_acc, step=step)
        utils.write_acc_loss(tb_writer, prefix='train', loss=loss, acc=acc, step=step)
        lr_summary = tf.Summary(value=[tf.Summary.Value(tag='learning_rate', simple_value=float(lr))])
        tb_writer.add_summary(lr_summary, step)
        print 'train acc :{:06.4f} train loss : {:06.4f} val acc : {:06.4f} val loss : {:06.4f}'.format(acc, loss,
                                                                                                        val_acc,
                                                                                                        val_cost)
