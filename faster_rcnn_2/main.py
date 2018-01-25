#-*- coding:utf-8 -*-
import sys
sys.path.insert(0, '../')
from cnn import convolution2d , affine
import os
import os.path as osp
from tqdm import tqdm, trange
from scipy.misc import imread
import numpy as np
from distutils import spawn
import tensorflow as tf
from configure import cfg
import anchor_target_layer
import proposal_layer
import proposal_target_layer
import roi_pool
import loss_functions
import image_preprocessing

class FasterRcnnConv5():
    def __init__(self , n_classes  , eval_mode , data_dir):
        self.n_classes  = n_classes
        self.eval_mode = eval_mode
        self.data_dir = data_dir
        self._read_names()
        self.x_ = tf.placeholder(dtype=tf.float32 , shape = [1 , None , None ,1 ])
        self.im_dims = tf.placeholder(tf.int32 , [None ,2 ])
        self.gt_boxes = tf.placeholder(tf.int32 , [None ,5 ])

        self.step=0
        self._build()
        self._start_session()


    def _read_names(self):
        self.train_name_path=os.path.join(self.data_dir , 'Names' , 'train.txt')
        self.test_name_path = os.path.join(self.data_dir, 'Names', 'test.txt')
        self.val_name_path = os.path.join(self.data_dir, 'Names', 'valid.txt')

        self.train_names = [line.rstrip() for line in open(self.train_name_path ,'r')]
        self.val_names = [line.rstrip() for line in open(self.val_name_path, 'r')]
        self.test_names = [line.rstrip() for line in open(self.test_name_path, 'r')]


        print 'the Number of Training {}'.format(len(self.train_names))
        print 'the Number of Validation {}'.format(len(self.val_names))
        print 'the Number of Test {}'.format(len(self.test_names))



    def _build(self):
        self._convnet() #1.convolution 5 layer
        self._rpn()
        self._roi_proposal()
        self._fast_rcnn()
        self._optimizer()



    def _convnet(self):
        print '###### Convolution Network building.... '
        print
        kernels=[5, 3, 3, 3, 3]
        out_channels=[32, 64, 64, 128, 128]
        strides = [2, 2, 1, 2, 1]
        layer=self.x_
        for i in range(5):
            layer = convolution2d(name='conv_{}'.format(i), x=layer, out_ch=out_channels[i], k=kernels[i], s=strides[i],
                                  padding='SAME')
        self.top_conv = tf.identity(layer , 'top_conv')
        self._feat_stride = np.prod(strides)
        print

    def _rpn(self):
        print '###### Region Proposal Network building.... '
        print
        """
        , RPN predicts the possibility of an anchor being background or foreground, and refine the anchor.
        :return:
        """
        rpn_out_ch = 512
        rpn_k=3
        anchor_scales = [1, 2, 4]  # original anchor_scales
        n_anchors = len(anchor_scales) * 3 # len(ratio) =3
        #_n_anchors =len(self.anchor_scales)*3
        top_conv = self.top_conv
        with tf.variable_scope('rpn'):
            self.rpn_layer = convolution2d('conv', top_conv, out_ch= rpn_out_ch , k=rpn_k  , s=1 ,padding="SAME") #shape=(1, ?, ?, 512)
            with tf.variable_scope('cls'):
                    rpn_cls_layer = convolution2d('conv' ,self.rpn_layer, out_ch= n_anchors*2 , k=1 , act=None , s=1)
                    self.rpn_cls_layer  = tf.identity(rpn_cls_layer , name='cls_output')
                    print '** cls layer shape : {}'.format(np.shape(rpn_cls_layer )) #(1, ?, ?, 18)
            with tf.variable_scope('target'):
                """
                print layer
                print self.gt_boxes
                print self.im_dims
                print self._feat_stride
                print anchor_scales
                
                __C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
                # Give the positive RPN examples weight of p * 1 / {num positives}
                # and give negatives a weight of (1 - p)
                # Set to -1.0 to use uniform example weighting

                """
                self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights = anchor_target_layer.anchor_target_layer(
                    rpn_cls_score=self.rpn_cls_layer, gt_boxes=self.gt_boxes, im_dims=self.im_dims,
                    _feat_stride=self._feat_stride, anchor_scales=anchor_scales)
                # layer shape : 1 ? ? 18
                # gt.boxes placeholder : ? ,5
                # img_dim : ? 2
            with tf.variable_scope('bbox'):
                # Bounding-Box regression layer (bounding box predictions)
                rpn_bbox_layer = convolution2d('conv', self.rpn_layer, out_ch=n_anchors * 4, k=1, s=1,
                                               act=None)  # (1, ?, ?, 36)
            self.rpn_bbox_layer = tf.identity(rpn_bbox_layer , 'bbox_output')
            print '** bbox_layer shape : {}'.format(np.shape(self.rpn_bbox_layer ))
            print
    def _roi_proposal(self):
        print '###### ROI Proposal Network building.... '
        print

        self.num_classes = cfg.NUM_CLASSES #1 background or Target
        self.anchor_scales = cfg.RPN_ANCHOR_SCALES # [8, 16, 32]
        self.rpn_cls_prob=self._rpn_softmax()
        key = 'TRAIN' if self.eval_mode is False else 'TEST'
        self.blobs =proposal_layer.proposal_layer(rpn_bbox_cls_prob=self.rpn_cls_prob , rpn_bbox_pred=self.rpn_bbox_layer,
                                    im_dims=self.im_dims, cfg_key=key, _feat_stride=self._feat_stride,
                                    anchor_scales=self.anchor_scales)
        if self.eval_mode is False:
            # Calculate targets for proposals
            self.rois, self.labels, self.bbox_targets, self.bbox_inside_weights, self.bbox_outside_weights = \
                proposal_target_layer.proposal_target_layer(rpn_rois=self.blobs, gt_boxes=self.gt_boxes,
                                      _num_classes=self.num_classes)
    def _fast_rcnn(self):
        print '###### Fast R-CNN building.... '
        print
        with tf.variable_scope('fast_rcnn'):
            keep_prob = cfg.FRCNN_DROPOUT_KEEP_RATE if self.eval_mode is False else 1.0
            pooledFeatures = roi_pool.roi_pool(self.top_conv, self.rois, self.im_dims) #roi pooling
            layer = pooledFeatures # ? 7,7 128 Same Output
            #print layer
            for i in range(len(cfg.FRCNN_FC_HIDDEN)):
                layer = affine('fc_{}'.format(i) , layer ,cfg.FRCNN_FC_HIDDEN[i])
            with tf.variable_scope('cls'):
                self.fast_rcnn_cls_logits = affine('cls_logits' , layer , self.num_classes ,activation=None)
            with tf.variable_scope('bbox'):
                self.fast_rcnn_bbox_logits = affine('bbox_logits' , layer , self.num_classes*4,activation=None)


    def _optimizer(self):

        self.lr=0.0001
        self.step=0
        # rpn optimzer
        self.rpn_cls_loss=loss_functions.rpn_cls_loss(self.rpn_cls_layer,self.rpn_labels)
        self.rpn_bbox_loss = loss_functions.rpn_bbox_loss(rpn_bbox_pred=self.rpn_bbox_layer,
                                                          rpn_bbox_targets=self.rpn_bbox_targets,
                                                          rpn_inside_weights=self.rpn_bbox_inside_weights,
                                                          rpn_outside_weights=self.rpn_bbox_outside_weights)


        # fast-rcnn optimzer
        self.fast_rcnn_cls_loss=loss_functions.fast_rcnn_cls_loss(self.fast_rcnn_cls_logits, self.labels)
        self.fast_rcnn_bbox_loss = loss_functions.fast_rcnn_bbox_loss(fast_rcnn_bbox_pred=self.fast_rcnn_cls_logits,
                                                                      bbox_targets=self.bbox_targets,
                                                                      roi_inside_weights=self.bbox_inside_weights,
                                                                      roi_outside_weights=self.bbox_outside_weights)





        self.cost=tf.reduce_sum(self.rpn_cls_loss + self.rpn_bbox_loss +self.fast_rcnn_cls_loss + self.fast_rcnn_bbox_loss)
        decay_steps = cfg.TRAIN.LEARNING_RATE_DECAY_RATE * len(self.train_names)  # Number of Epochs x images/epoch
        learning_rate = tf.train.exponential_decay(learning_rate=self.lr, global_step=self.step,
                                                   decay_steps=decay_steps, decay_rate=cfg.TRAIN.LEARNING_RATE_DECAY,
                                                   staircase=True)
        # Optimizer: ADAM
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        #cost = rpn_cls_loss+ rpn_bbox_loss + fast_rcnn_cls_loss + fast_rcnn_bbox_loss

    def _rpn_softmax(self):
        shape=tf.shape(self.rpn_cls_layer)
        rpn_cls_score = tf.transpose(self.rpn_cls_layer,[0,3,1,2]) # Tensor("transpose:0", shape=(1, 18, ?, ?)
        rpn_cls_score = tf.reshape(rpn_cls_score,[shape[0],2,shape[3]//2*shape[1],shape[2]])# Tensor("transpose:0", shape=(1, 2, ?, ?)
        rpn_cls_score = tf.transpose(rpn_cls_score,[0,2,3,1]) #Tensor("transpose_1:0", shape=(?, ?, ?, 2), dtype=float32)

        self.rpn_cls_prob_ori = tf.nn.softmax(rpn_cls_score)#Tensor("transpose_1:0", shape=(?, ?, ?, 2), dtype=float32)
        # Reshape back to the original
        rpn_cls_prob = tf.transpose(self.rpn_cls_prob_ori, [0, 3, 1, 2]) #Tensor("transpose_2:0", shape=(?, 2, ?, ?), dtype=float32)
        rpn_cls_prob = tf.reshape(rpn_cls_prob, [shape[0], shape[3], shape[1], shape[2]]) #Tensor("transpose_2:0", shape=(?, ?, ?, ?), dtype=float32)
        rpn_cls_prob = tf.transpose(rpn_cls_prob, [0, 2, 3, 1])#Tensor("transpose_2:0", shape=(?, ?, ?, ?), dtype=float32)

        return rpn_cls_prob

    def flatten(self, input ,keep_prob=1):
        """
        Flattens 4D Tensor (from Conv Layer) into 2D Tensor (to FC Layer)
        :param keep_prob: int. set to 1 for no dropout
        """
        self.count['flat'] += 1
        scope = 'flat_' + str(self.count['flat'])
        with tf.variable_scope(scope):
            # Reshape function
            input_nodes = tf.Dimension(
                input.get_shape()[1] * input.get_shape()[2] * input.get_shape()[3])
            output_shape = tf.stack([-1, input_nodes])
            input = tf.reshape(input, output_shape)
            # Dropout function
            if keep_prob != 1:
                input = tf.nn.dropout(input, keep_prob=keep_prob)
        return input
        self.print_log(scope + ' output: ' + str(input.get_shape()))

    def train(self , file_epoch):
        train_order = np.random.permutation(len(self.train_names))
        self.file_epoch=file_epoch
        tf_inputs = (self.x_, self.im_dims, self.gt_boxes)
        self.step +=1
        print self.step
        for self.epoch in trange(1, self.file_epoch + 1, desc='epochs'):
            for i in tqdm(train_order):
                feed_dict=self._create_feed_dict_for_train(i)
                try:
                    _,loss ,cls_prob= self.sess.run([self.optimizer,self.cost , self.rpn_cls_prob_ori], feed_dict=feed_dict)

                    #print 'loss',loss
                    #print 'cls_prob', cls_prob
                    #print np.shape(cls_prob)
                except Exception as e:
                    print e
                    pass;

    def _create_feed_dict_for_train(self , image_idx):
        img_path=os.path.join(self.data_dir , 'Images' ,self.train_names[image_idx]+cfg.IMAGE_FORMAT  )
        annotation_path = os.path.join(self.data_dir, 'Annotations', self.train_names[image_idx] + '.txt')
        img = imread(img_path)

        gt_bbox = np.loadtxt(annotation_path , ndmin=2)
        im_dims = np.array(img.shape[:2]).reshape([1,2])

        #print 'gt_boxx',gt_bbox
        #print 'image dimension',im_dims
        flips = [0, 0]
        flips[0] = np.random.binomial(1,0.5)
        img = image_preprocessing.image_preprocessing(img)
        if np.max(img) > 1 :
            img=img/255.
        feed_dict = {self.x_: img, self.im_dims: im_dims, self.gt_boxes : gt_bbox}
        return feed_dict


    def _start_session(self):
        config = tf.ConfigProto(log_device_placement=False)
        #config.gpu_options.per_process_gpu_memory_fraction = vram
        self.sess = tf.Session()
        init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
        self.sess.run(init)


if __name__ == '__main__':
    data_dir ='./clutteredMNIST'
    model=FasterRcnnConv5(10 , eval_mode=False , data_dir=data_dir)
    model.train(1)


