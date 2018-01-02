#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle
import os
import utils
import data
import glob
from cnn import affine , algorithm ,lr_schedule
from PIL import Image
from sklearn.decomposition import PCA
import PIL

"""saved pre trained list 
 1.inception v3 model """

inception_v3_url= "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"


def download_and_extract_model(url , data_dir):

    utils.donwload(url, download_dir= data_dir)
    name=url.split('/')[-1]
    file_path=os.path.join(data_dir,name)
    utils.extract(file_path,data_dir)


#dense net 처럼 다시 설계해야 한다
"""
class Transfer
    |
    |-inception_v3 
    |-resnet
    |-inception_v5
    |-vgg net
"""

class Transfer_inception_v3(object):
    """
    conv1 --> conv2 --> -->

    """
    def __init__(self , data_dir):

        # inception
        x_jpeg_name = "DecodeJpeg/contents:0"  # input image for jpeg format
        x_name = "DecodeJpeg:0"  # input image for the other format
        x_resized_image = "ResizeBilinear:0"
        softmax = "softmax:0"
        logits = "softmax/logits:0"
        transfer_layer = "pool_3:0"

        # load graph ,variable and initialize session
        self.graph = tf.Graph() #make graph
        with self.graph.as_default() :#set graph to default
            #만약 여기서 with 을 주지 않으면 , 아래 그래프을 복원하는 것들은 실행 되지 않는다.
            self.data_dir=data_dir
            pb_path_list=glob.glob(os.path.join(data_dir , '*.pb'))
            assert len(pb_path_list) ==1 , 'the number of protobuffer has to be 1 , {}'.format(len(pb_path_list))
            pb_path=pb_path_list[0]
            print 'Proto Buffer file path - {}'.format(pb_path)

            gfile = tf.gfile.FastGFile(pb_path, 'rb') #get file point ,
            graph_def = tf.GraphDef() # make graph definition,
            graph_def.ParseFromString(gfile.read())  # load pb file to into graph_def
            tf.import_graph_def(graph_def, name='')  # import graph_def into tensorflow graph

            self.x_ = tf.get_default_graph().get_tensor_by_name(x_name)
            # Get tensor name from inception v3 graph
            self.pred = tf.get_default_graph().get_tensor_by_name(softmax)
            self.logits = tf.get_default_graph().get_tensor_by_name(logits)
            self.resized_image = tf.get_default_graph().get_tensor_by_name(x_resized_image)
            self.transfer_layer = tf.get_default_graph().get_tensor_by_name(transfer_layer)
            self.sess= tf.Session(graph=self.graph) # import graph to session
            self.transfer_layer_len=self.transfer_layer.get_shape()[3] #transfer layer shape : [1,1,1,2048]




        #feed dict
        # Image is passed in as a 4-dimension
        # The pixels MUST be values between 0 and 255 (float or int).


    def _create_feed_dict(self , image ):
        feed_dict = {self.x_: image}
        return feed_dict

    def classify(self , image ):
        assert np.max(image) > 1 , 'max values of images {} '.format(np.max(image))
        feed_dict=self._create_feed_dict(image)
        pred = self.sess.run(fetches=self.pred , feed_dict= feed_dict)
        print 'pred shape : {}'.format(np.shape(pred))
        pred = np.squeeze(pred)
        return pred

    def resize_image(self , image ):
        assert np.max(image) > 1, 'max values of images {} '.format(np.max(image))
        feed_dict = self._create_feed_dict(image)
        resized_image = self.sess.run(fetches=self.resized_image, feed_dict=feed_dict)
        resized_image = resized_image.squeeze(axis=0)

    def normalize_image(self , image):
        assert np.max(image) > 1, 'max values of images {} '.format(np.max(image))
        return image/255.

    def get_transfer_values(self , image ):
        assert np.max(image) > 1, 'max values of images {} '.format(np.max(image))
        # Get transfer value from pooling layer
        feed_dict=self._create_feed_dict(image)
        transfer_values = self.sess.run(fetches=self.transfer_layer , feed_dict = feed_dict )
        transfer_values=np.squeeze(transfer_values)
        return transfer_values
    def images_to_transfer_values(self , images):
        n_images=len(images)
        multiple_values=map(lambda image : self.get_transfer_values(image) , images )
        multiple_values=np.asarray(multiple_values)
        print 'multiple values shape {}'.format(np.shape(multiple_values))
        return multiple_values

    def images2caches(self ,cache_path , images):
        if os.path.exists(cache_path):
            with open(cache_path, mode='rb') as file:
                obj = pickle.load(file)
        else:
            with open(cache_path, mode='wb') as file:
                obj = self.images_to_transfer_values(images)
                pickle.dump(obj, file)
            print("- Data saved to cache-file: " + cache_path)
        return obj

if __name__ =='__main__':

    download_and_extract_model(url=inception_v3_url , data_dir='./pretrained_models/inception_v3')
    ckpt_dir = 'inception_v3_pretrained'
    train_imgs, train_labs, train_filenames, test_imgs, test_labs, test_filenames = data.type2('./fundus_300',
                                                                                               save_dir_name=ckpt_dir)
    n_classes=2
    sample_img=Image.open('./pretrained_models/inception_v3/cropped_panda.jpg')
    sample_img=np.asarray(sample_img)
    sample_imgs=np.vstack((sample_img.reshape([1,100,100,3]), sample_img.reshape([1,100,100,3])))


    print np.shape(sample_img)
    print np.shape(sample_imgs)

    model=Transfer_inception_v3(data_dir='./pretrained_models/inception_v3')
    values=model.get_transfer_values(sample_img)
    train_imgs = model.images2caches('pretrained_models/inception_v3/train_cache.pkl', train_imgs)
    test_imgs = model.images2caches('pretrained_models/inception_v3/test_cache.pkl', test_imgs)

    print 'training data shape : {}'.format(np.shape(train_imgs))
    print 'test data shape : {}'.format(np.shape(test_imgs))

    x_ = tf.placeholder(dtype=tf.float32 , shape = [None , 2048])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y_')
    phase_train = tf.placeholder(dtype =tf.bool , name='phase_train')
    lr_ = tf.placeholder(dtype=tf.float32, name='learning_rate')

    logits=affine('fc',x_ , out_ch= n_classes)
    pred, pred_cls, cost, train_op, correct_pred, accuracy = algorithm(logits, y_=y_, learning_rate=lr_,
                                                                       optimizer='sgd', use_l2_loss=False)

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
    logs_path = os.path.join('./logs', 'fundus_resnet', ckpt_dir)
    tb_writer = tf.summary.FileWriter(logs_path)
    tb_writer.add_graph(tf.get_default_graph())
    best_acc_ckpt_dir = os.path.join('./model', ckpt_dir, 'best_acc')
    best_loss_ckpt_dir = os.path.join('./model', ckpt_dir, 'best_loss')
    last_model_ckpt_dir = os.path.join('./model', ckpt_dir, 'last_model')
    last_model_ckpt_path = os.path.join(last_model_ckpt_dir, 'model')
    try:
        os.makedirs(last_model_ckpt_dir)
    except Exception as e:
        pass;
    start_step = utils.restore_model(saver=last_model_saver, sess=sess, ckpt_dir=last_model_ckpt_dir)




    """----------------------------------------------------------------------------------------------------------------
                                                    Training Model                                  
    ----------------------------------------------------------------------------------------------------------------"""
    batch_size=60
    lr_iters = [2000, 50000]
    lr_values=[0.0007 , 0.0001]
    max_acc, min_loss = 0, 10000000
    max_iter=1000000
    for step in range( start_step, max_iter):
        lr = lr_schedule(step , lr_iters , lr_values)
        batch_xs, batch_ys = data.next_batch(train_imgs, train_labs, batch_size=batch_size)
        _, loss, acc = sess.run(fetches=[train_op, cost, accuracy],
                                feed_dict={x_: batch_xs, y_: batch_ys, phase_train: True, lr_: lr})
        last_model_saver.save(sess, save_path=last_model_ckpt_path, global_step=step)
        if step % 100 == 0:

            # Get Validation Accuracy and Loss
            pred_list, cost_list = [], []
            val_pred, val_cost = sess.run(fetches=[pred, cost],
                                              feed_dict={x_: batch_xs, y_: batch_ys, phase_train: False})
            val_acc = utils.get_acc(val_pred, test_labs)
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











