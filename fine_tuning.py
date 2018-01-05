#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
import utils
import os
from cnn import gap , algorithm
import argparse
import data
import aug
parser=argparse.ArgumentParser()
parser.add_argument('--ckpt_dir' , type=str)
parser.add_argument('--batch_size' , type=int , default=10)
args=parser.parse_args()


"""
The difference between Transfer Learning and Fine-Tuning is that in Transfer Learning we only optimize the weights of
 the new classification layers we have added, while we keep the weights of the original VGG16 model.
 we keep the weights of the original VGG16 model.
 
 -saved model
 
 class fine_tuning 
    |- alexnet 
    |- resnet
    |- vgg16
    |-
    
 each pretrained_class
    |- download 
    |- 
    
"""


class fine_tuning(object):
    def __init__(self , model_name):
        self.model_name = model_name

    def _build_models(self):
        if self.model_name =='vgg16':
            model=vgg_16('./pretrained_model/vgg16')
        elif self.model_name == 'inception_v3':
            model = ('./pretrained_model/vgg16')
        else:
            raise AssertionError


class vgg_16(object):
    def __init__(self , n_classes , optimizer , input_shape , use_l2_loss ): # pb  = ProtoBuffer
        self.input_shape = input_shape #input shape = ( h ,w, ch )
        self.img_h,self.img_w,self.img_ch = self.input_shape
        self.n_classes=n_classes
        self.optimizer = optimizer  # build_model에서 사용된다
        self.use_l2_loss = use_l2_loss

        self.vgg16_pretrained_data_url = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"
        self.data_dir = 'pretrained_models/vgg_16'
        self.name_pb = 'vgg16.tfmodel'
        self.images_ = "images:0"
        self.dropout_name = 'dropout/random_uniform:0'
        self.dropout_1_name = 'dropout_1/random_uniform:0'
        self.layer_names = ['conv1_1/conv1_1', 'conv1_2/conv1_2',
                       'conv2_1/conv2_1', 'conv2_2/conv2_2',
                       'conv3_1/conv3_1', 'conv3_2/conv3_2', 'conv3_3/conv3_3',
                       'conv4_1/conv4_1', 'conv4_2/conv4_2', 'conv4_3/conv4_3',
                       'conv5_1/conv5_1', 'conv5_2/conv5_2', 'conv5_3/conv5_3']
        self.path_pb = os.path.join(self.data_dir , self.name_pb)
        if not os.path.exists(os.path.join(self.data_dir , self.name_pb)):
            utils.donwload(self.vgg16_pretrained_data_url  ,download_dir=self.data_dir)
        self.graph=tf.Graph()
        with self.graph.as_default():
            """
            TensorFlow graphs are saved to disk as so-called Protocol Buffers
            # aka. proto-bufs which is a file-format that works on multiple
            # platforms. In this case it is saved as a binary file.
            """
            gfile=tf.gfile.FastGFile(self.path_pb, 'rb')
            graph_def = tf.GraphDef();
            graph_def.ParseFromString(gfile.read())
            tf.import_graph_def(graph_def ,name='')
            """------------------------------------------------------------------------------
                                                build model
            -------------------------------------------------------------------------------"""
            self.sess = tf.Session(graph=self.graph)
            self._reconstruct_layers()
            self._build_model()
            """------------------------------------------------------------------------------
                                                session setting
            -------------------------------------------------------------------------------"""
            self.saver = tf.train.Saver(max_to_keep=10000000)
            self.last_model_saver = tf.train.Saver(max_to_keep=1)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = False
            self.sess = tf.Session(graph=self.graph , config=config)
            init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
            self.sess.run(init)


    def _reconstruct_layers(self):
        """------------------------------------------------------------------------------
                 Naming rule :
                     conv1_1/w:0 ~ conv5_3/w:0
                     conv1_1/b:0 ~ conv5_3/b:0
        -------------------------------------------------------------------------------"""
        self.weights_list = []
        self.biases_list = []
        print "trying reconstruct weights and biases..."
        for i, name in enumerate(self.layer_names):
            utils.show_progress(i, len(self.layer_names))
            conv_name = name.replace(":0", '')
            name = name.split('/')[0]
            w_name = name + '/filter:0'
            b_name = name + '/biases:0'

            w_, b_ = self.sess.run([w_name, b_name])
            self.weights_list.append(tf.Variable(w_, name=w_name.replace("filter:0", 'w')))
            self.biases_list.append(tf.Variable(b_, name=b_name.replace("biases:0", 'b')))


    def _build_model(self):
        """------------------------------------------------------------------------------
                                        Input Data
        -------------------------------------------------------------------------------"""

        self.x_ = tf.placeholder(dtype = tf.float32 , shape = [None , self.img_h , self.img_w , self.img_ch ])
        self.y_ = tf.placeholder(dtype=tf.int32, shape=[None, self.n_classes], name='y_')
        self.lr_ = tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.phase_trin = tf.placeholder(dtype=tf.bool)
        layer=self.x_
        # data augmentation
        """------------------------------------------------------------------------------
                                        VGG 16 network
        -------------------------------------------------------------------------------"""
        max_pool_idx=[1,3,6,9,12]
        for i in range(len(self.weights_list)):
            w=self.weights_list[i]
            b=self.biases_list[i]
            with tf.variable_scope('layer_'+str(i)):
                conv_name=w.name.split('/')[0]
                layer=tf.nn.conv2d(layer ,w , strides=[1,1,1,1] , padding='SAME' , name=conv_name) + b
                layer=tf.nn.relu(layer , name='activation')
                if i in max_pool_idx:
                    layer=tf.nn.max_pool(layer , ksize=[1,2,2,1] ,strides=[1,2,2,1], padding ='SAME' , name='pool')
        top_conv = tf.identity(layer, 'top_conv')
        self.logits=gap('gap' , top_conv , self.n_classes)
        self.pred, self.pred_cls, self.cost, self.train_op, self.correct_pred, self.accuracy = algorithm(self.logits,
                                                                                                         self.y_,
                                                                                                         self.lr_,
                                                                                                         self.optimizer,
                                                                                                         self.use_l2_loss)
if '__main__' == __name__ :

    #image, label = utils.read_one_example('./fundus_300_debug/debug_cataract_glaucoma_test.tfrecord',(299, 299))
    train_imgs, train_labs, train_filenames, test_imgs, test_labs, test_filenames = data.type2('./fundus_300',
                                                                                               save_dir_name=args.ckpt_dir)
    test_imgs_list, test_labs_list = utils.divide_images_labels_from_batch(test_imgs, test_labs, batch_size=60)
    test_imgs_labs = zip(test_imgs_list, test_labs_list)

    train_imgs=train_imgs/255.
    test_imgs = test_imgs/255.
    model=vgg_16(n_classes=2 , optimizer='sgd' , input_shape=(299,299,3) ,use_l2_loss=True)
    """------------------------------------------------------------------------------
                                        Dir Setting                    
    -------------------------------------------------------------------------------"""
    logs_path = os.path.join('./logs', 'fundus_fine_tuning', args.ckpt_dir)
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
    start_step = utils.restore_model(saver=model.last_model_saver, sess=model.sess, ckpt_dir=last_model_ckpt_dir)
    max_acc, min_loss = 0, 10000000
    max_iter=10000000
    for step in range(start_step , max_iter):
        utils.show_progress(step , max_iter)
        batch_xs, batch_ys = data.next_batch(train_imgs, train_labs, batch_size=args.batch_size)
        rotate_imgs = map(lambda batch_x: aug.random_rotate(batch_x), batch_xs)
        _, loss, acc = model.sess.run(fetches=[model.train_op, model.cost, model.accuracy],
                                feed_dict={model.x_: batch_xs, model.y_: batch_ys,  model.lr_: 0.0001})
        model.last_model_saver.save(model.sess, save_path=last_model_ckpt_path, global_step=step)
        if step % 100 ==0:
            pred_list, cost_list = [], []
            for batch_xs, batch_ys in test_imgs_labs:
                batch_pred, batch_cost = model.sess.run(fetches=[model.pred, model.cost],
                                                  feed_dict={model.x_: batch_xs, model.y_: batch_ys, })
                pred_list.extend(batch_pred)
                cost_list.append(batch_cost)
            val_acc = utils.get_acc(pred_list, test_labs)
            val_cost = np.sum(cost_list) / float(len(cost_list))
            max_acc, min_loss = utils.save_model(model.sess, model.saver, max_acc, min_loss, val_acc, val_cost, best_acc_ckpt_dir,
                                                 best_loss_ckpt_dir,
                                                 step)
            utils.write_acc_loss(tb_writer, prefix='test', loss=val_cost, acc=val_acc, step=step)
            utils.write_acc_loss(tb_writer, prefix='train', loss=loss, acc=acc, step=step)
            #lr_summary = tf.Summary(value=[tf.Summary.Value(tag='learning_rate', simple_value=float(lr))])
            #tb_writer.add_summary(lr_summary, step)
            print 'train acc :{:06.4f} train loss : {:06.4f} val acc : {:06.4f} val loss : {:06.4f}'.format(acc, loss,
                                                                                                            val_acc,
                                                                                                            val_cost)
