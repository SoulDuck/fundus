#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
import utils
import os
from cnn import gap , algorithm , lr_schedule , affine , dropout
import argparse
import data
import aug
from cifar_ import input as cifar_input


parser=argparse.ArgumentParser()
parser.add_argument('--ckpt_dir' , type=str ,default='finetuning_vgg_16_cifar' ) #default='finetuning_vgg_16'
parser.add_argument('--batch_size' , type=int , default=40)
parser.add_argument('--lr_iters' ,nargs='+', type=int, default=[5000 ,15000 , 40000 , 80000] )
parser.add_argument('--lr_values',nargs='+', type=float, default=[0.0001 , 0.00007 , 0.00004 , 0.00001])
parser.add_argument('--logits_type', type=str, default='fc')
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




class Transfer_vgg_16(object):
    def __init__(self, n_classes, optimizer, input_shape, use_l2_loss, img_size_cropped, color_aug):  # pb  = ProtoBuffer
        self.input_shape = input_shape #input shape = ( h ,w, ch )
        self.img_h,self.img_w,self.img_ch = self.input_shape
        self.n_classes=n_classes
        self.optimizer = optimizer  # build_model에서 사용된다
        self.use_l2_loss = use_l2_loss

        self.weights_saved_dir=os.path.join('pretrained_models' , 'vgg_16' , 'model_weights') #
        self.img_size_cropped  = img_size_cropped
        self.color_aug = color_aug
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
            self._save_pretrained_weights()
        tf.reset_default_graph() # reset pretrained train graph and load saved weights and re-draw graph
        self._load_pretrained_weights()
        self._build_model()
        """------------------------------------------------------------------------------
                                            session setting
        -------------------------------------------------------------------------------"""
        self.saver = tf.train.Saver(max_to_keep=10000000)
        self.last_model_saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
        self.sess.run(init)

    def _save_pretrained_weights(self):
        """------------------------------------------------------------------------------
                 Naming rule :
                     original weights name : conv1_1/conv1_1/filters:0
                     changed weights name  : conv1_1/conv1_1_w

                     conv1_1/w:0 ~ conv5_3/_w:0
                     conv1_1/b:0 ~ conv5_3/_b:0
        -------------------------------------------------------------------------------"""
        print "trying reconstruct weights and biases..."
        for i, name in enumerate(self.layer_names):
            #customizing 한 weights 와 biases 들을 구별할려고 filters -->  w  \ biases -->b 로 바꿨당
            utils.show_progress(i, len(self.layer_names))
            conv_name = name.replace(":0", '')
            name = name.split('/')[0] # conv_1/conv_1 -->conv_1
            w_name = name + '/filter:0'
            b_name = name + '/biases:0'

            w_, b_ = self.sess.run([w_name, b_name])
            w_name = w_name.replace("/filter:0", '_w')
            b_name = b_name.replace("/biases:0", '_b')
            w_path=os.path.join(self.weights_saved_dir, w_name+'.npy')
            b_path=os.path.join(self.weights_saved_dir, b_name+'.npy')
            if os.path.exists(w_path) and os.path.exists(b_path):
                print 'weight {} , biases {} already exist'.format(w_name , b_name )
                continue;
            else:
                np.save(os.path.join(self.weights_saved_dir,w_name),w_) #conv filter save
                np.save( os.path.join(self.weights_saved_dir, b_name),b_) #conv biases save
        print 'save complete!'

    def _load_pretrained_weights(self ):
        self.weights_list=[]
        self.biases_list=[]
        for i , name in enumerate(self.layer_names):
            utils.show_progress(i, len(self.layer_names))
            name = name.split('/')[0]
            w_path = os.path.join(self.weights_saved_dir, name +'_w.npy')
            b_path = os.path.join(self.weights_saved_dir, name+ '_b.npy')
            w = np.load(w_path)
            b = np.load(b_path)
            self.weights_list.append(w)
            self.biases_list.append(b)

    def _build_model(self):
        """------------------------------------------------------------------------------
                                        Input Data
        -------------------------------------------------------------------------------"""
        self.x_ = tf.placeholder(dtype = tf.float32 , shape = [None , self.img_h , self.img_w , self.img_ch ])
        self.y_ = tf.placeholder(dtype=tf.int32, shape=[None, self.n_classes], name='y_')
        self.lr_ = tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.phase_train = tf.placeholder(dtype=tf.bool)
        # data augmentation
        """------------------------------------------------------------------------------
                                        Build Up VGG 16 network
        -------------------------------------------------------------------------------"""
        max_pool_idx=[1,3,6,9,12]
        layer = aug.aug_tensor_images(self.x_, phase_train=self.phase_train, img_size_cropped=self.img_size_cropped,
                                      color_aug=self.color_aug)
        tl_flag=True # TL = Transfer_learning
        for i , name in enumerate(self.layer_names):
            w=self.weights_list[i]
            b=self.biases_list[i]
            #print np.shape(w)
            conv_name=name.split('/')[0] # conv1_1
            with tf.variable_scope(conv_name):
                w_name=os.path.join(conv_name , 'filters') # /conv1_1/filters
                b_name = os.path.join(conv_name, 'biases')
                w = tf.Variable(w , name=w_name , trainable=False) #frozen convolution weight
                b = tf.Variable(b, name=b_name, trainable=False) #frozen convolution weight
                layer=tf.nn.conv2d(layer ,w , strides=[1,1,1,1] , padding='SAME' , name=conv_name) + b
                layer=tf.nn.relu(layer , name='activation')
                if i in max_pool_idx:
                    layer=tf.nn.max_pool(layer , ksize=[1,2,2,1] ,strides=[1,2,2,1], padding ='SAME' , name='pool')
        top_conv = tf.identity(layer, 'top_conv')
        print 'Logits type : {}'.format(args.logits_type)
        if args.logits_type=='gap':
            self.logits=gap('gap' , top_conv , self.n_classes)
        elif args.logits_type=='fc':
            fc_filters=[4096 , 4096]
            for i,out_ch in enumerate(fc_filters):
                layer=affine('fc_{}'.format(i) , layer, out_ch)
                layer=dropout(layer , phase_train=self.phase_train , keep_prob=0.5)
                print 'fc_{} dropout applied'.format(i)
            self.logits=affine('logits'.format(i) , layer, self.n_classes,activation=None)
        self.pred, self.pred_cls, self.cost, self.train_op, self.correct_pred, self.accuracy = algorithm(self.logits,
                                                                                                         self.y_,
                                                                                                         self.lr_,
                                                                                                         self.optimizer,
                                                                                                         self.use_l2_loss)

class FineTuning_vgg_16(Transfer_vgg_16):
    def __init__(self , model_dir , logits_type , weight_saved_dir):
        self.logits_type = logits_type
        self.model_dir = model_dir
        # create Session
        self.sess = tf.Session()
        # restore model
        self._restore_best_model()
        # save parameters
        self._save_pretrained_weights()


        self._build_models()
        self._restore_best_model()
        self.weight_saved_dir = weight_saved_dir
        self.layer_names = ['conv1_1/conv1_1', 'conv1_2/conv1_2',
                            'conv2_1/conv2_1', 'conv2_2/conv2_2',
                            'conv3_1/conv3_1', 'conv3_2/conv3_2', 'conv3_3/conv3_3',
                            'conv4_1/conv4_1', 'conv4_2/conv4_2', 'conv4_3/conv4_3',
                            'conv5_1/conv5_1', 'conv5_2/conv5_2', 'conv5_3/conv5_3']

    def _restore_best_model(self):
        self.saver = tf.train.import_meta_graph(
            meta_graph_or_file=self.best_model_dirpath + '.meta')  # example model path ./models/fundus_300/5/model_1.ckpt
        self.saver.restore(self.sess, save_path=self.best_model_dirpath)  # example model path ./models/fundus_300/5/model_1.ckpt

    def _save_pretrained_weights(self):
        """------------------------------------------------------------------------------

                 Naming rule :
                     conv1_1/w:0 ~ conv5_3/w:0
                     conv1_1/b:0 ~ conv5_3/b:0
        -------------------------------------------------------------------------------"""
        print "trying restore weights and biases..."
        # convolution hpyer parameter save
        for i, name in enumerate(self.layer_names):
            #customizing 한 weights 와 biases 들을 구별할려고 filters -->  w  \ biases -->b 로 바꿨당
            utils.show_progress(i, len(self.layer_names))
            layer_name = layer_name.split('/')[0]
            w_name = layer_name + '/filter:0'
            b_name = layer_name + '/biases:0'

            w_, b_ = self.sess.run([w_name, b_name])
            w_name = w_name.replace("/filter:0", '_w')
            b_name = b_name.replace("/biases:0", '_b')
            np.save(os.path.join(self.weights_saved_dir,w_name),w_) #conv filter save
            np.save( os.path.join(self.weights_saved_dir, b_name),b_) #conv biases save
        # fully connected hyper parameter save
        if self.logits_type =='gap':
            w_name = os.path.join(self.logits_type + 'w').replace('/w:0' , '_w') #'gap/w' or 'gap/b'
            b_name = os.path.join(self.logits_type + 'b').replace('/b:0' , '_b')  # 'gap/w' or 'gap/b'
            w_ ,b_=self.sess.run(w_name , b_name)
            np.save(os.path.join(self.weights_saved_dir, w_name), w_)  # conv filter save
            np.save(os.path.join(self.weights_saved_dir, b_name), b_)  # conv biases save

        elif self.logits_type =='fc':
            # fc_0 --> fc_1--> logits
            count =0 # for counting the fully connected layer number
            while(True):
                try:
                    w_name = os.path.join(self.logits_type + '_{}'.format(count), 'w').replace('/w:0' , '_w')
                    # fc_0/w -->fc_0_w
                    b_name = os.path.join(self.logits_type + '_{}'.format(count), 'b').replace('/b:0' , '_b')
                    # fc_0/b --> fc_0_b
                    w_, b_ = self.sess.run(w_name, b_name)
                    np.save(os.path.join(self.weights_saved_dir, w_name), w_)  # conv filter save
                    np.save(os.path.join(self.weights_saved_dir, b_name), b_)  # conv biases save
                except Exception:
                    w_name = os.path.join('logits', 'w').replace('/w:0' , '_w')  # fc_0/w
                    b_name = os.path.join('logits', 'b').replace('/b:0' , '_b')  # fc_0/b
                    w_, b_ = self.sess.run(w_name, b_name)
                    np.save(os.path.join(self.weights_saved_dir, w_name), w_)  # conv filter save
                    np.save(os.path.join(self.weights_saved_dir, b_name), b_)  # conv biases save
                    break;
        else:
            raise NotImplementedError()

    def _load_pretrained_weights(self):
        self.conv_weights_list = []
        self.conv_biases_list = []
        self.fc_weights_list = []
        self.fc_biases_list = []
        for i, name in enumerate(self.layer_names):
            utils.show_progress(i, len(self.layer_names))
            name = name.split('/')[0]
            w_path = os.path.join(self.weights_saved_dir, name + '_w.npy')
            b_path = os.path.join(self.weights_saved_dir, name + '_b.npy')
            w = np.load(w_path)
            b = np.load(b_path)
            self.conv_weights_list.append(w)
            self.conv_biases_list.append(b)

        if self.logits_type == 'gap':
            w_path = os.path.join(self.weights_saved_dir, self.logits_type + '_w.npy')
            b_path = os.path.join(self.weights_saved_dir, self.logits_type + '_b.npy')
            w = np.load(w_path)
            b = np.load(b_path)
            self.conv_weights_list.append(w)
            self.conv_biases_list.append(b)
        elif self.logits_type == 'fc':
            count=0; # for counting the fully connected layer number
            while(True):
                w_path = os.path.join(self.weights_saved_dir, self.logits_type + '_{}'.format(count) + '_w.npy')
                #_save_pretrained_weights 에 저장된 형태를 불러오기 위해서.
                b_path = os.path.join(self.weights_saved_dir, self.logits_type + '_{}'.format(count) + '_b.npy')

                w = np.load(w_path)
                b = np.load(b_path)
                self.conv_weights_list.append(w)
                self.conv_biases_list.append(b)

            # fc_0 --> fc_1--> logits



if '__main__' == __name__ :


    #cifar-test
    train_imgs, train_labs, test_imgs, test_labs = cifar_input.get_cifar_images_labels(onehot=True,
                                                                                       data_dir='cifar_/cifar_10/cifar-10-batches-py')
    n,h,w,ch = np.shape(train_imgs)
    n,n_classes=np.shape(train_labs)

    #fundus-test
    """
    train_imgs, train_labs, train_filenames, test_imgs, test_labs, test_filenames = data.type2('./fundus_300_debug',
                                                                                               save_dir_name=args.ckpt_dir)
    """

    test_imgs_list, test_labs_list = utils.divide_images_labels_from_batch(test_imgs, test_labs, batch_size=60)
    test_imgs_labs = zip(test_imgs_list, test_labs_list)
    train_imgs=train_imgs/255.
    test_imgs = test_imgs/255.


    model = Transfer_vgg_16(n_classes=n_classes, optimizer='adam', input_shape=(h, w, ch), use_l2_loss=True,
                            img_size_cropped=h,
                            color_aug=False)

    """------------------------------------------------------------------------------
                                        Dir Setting                    
    -------------------------------------------------------------------------------"""
    logs_path = os.path.join('./logs', args.ckpt_dir)
    tb_writer = tf.summary.FileWriter(logs_path)
    tb_writer.add_graph(tf.get_default_graph())

    model_root_dir = os.path.join('./model' , args.ckpt_dir)
    last_model_ckpt_dir = os.path.join('./model', args.ckpt_dir, 'last')
    last_model_ckpt_path = os.path.join(last_model_ckpt_dir, 'model')
    try:
        os.makedirs(last_model_ckpt_dir)
    except Exception as e:
        pass;
    start_step = utils.restore_model(saver=model.last_model_saver, sess=model.sess, ckpt_dir=last_model_ckpt_dir)
    max_acc, min_loss = 0, 10000000
    max_iter=10000
    """------------------------------------------------------------------------------
                                Transfer Learning Stage                     
    -------------------------------------------------------------------------------"""

    for step in range(start_step , max_iter):
        lr = lr_schedule(step, args.lr_iters, args.lr_values)
        utils.show_progress(step , max_iter)
        batch_xs, batch_ys = data.next_batch(train_imgs, train_labs, batch_size=args.batch_size)
        rotate_imgs = map(lambda batch_x: aug.random_rotate(batch_x), batch_xs)
        #training
        _, loss, acc = model.sess.run(fetches=[model.train_op, model.cost, model.accuracy],
                                feed_dict={model.x_: batch_xs, model.y_: batch_ys,  model.lr_: lr , model.phase_train:True})
        #validation
        if step % 100 ==0:
            pred_list, cost_list = [], []
            for batch_xs, batch_ys in test_imgs_labs:
                batch_pred, batch_cost = model.sess.run(fetches=[model.pred, model.cost],
                                                  feed_dict={model.x_: batch_xs, model.y_: batch_ys,model.phase_train:False})
                pred_list.extend(batch_pred)
                cost_list.append(batch_cost)
            val_acc = utils.get_acc(pred_list, test_labs)
            val_cost = np.sum(cost_list) / float(len(cost_list))
            max_acc, min_loss = utils.save_model(model.sess, max_acc, min_loss, val_acc, val_cost, step, model_root_dir,
                                                 model.last_model_saver, model.saver)
            utils.write_acc_loss(tb_writer, prefix='test', loss=val_cost, acc=val_acc, step=step)
            utils.write_acc_loss(tb_writer, prefix='train', loss=loss, acc=acc, step=step)
            lr_summary = tf.Summary(value=[tf.Summary.Value(tag='learning_rate', simple_value=float(lr))])
            tb_writer.add_summary(lr_summary, step)
            print 'train acc :{:06.4f} train loss : {:06.4f} val acc : {:06.4f} val loss : {:06.4f}'.format(acc, loss,
                                                                                                            val_acc,
                                                                                                            val_cost)