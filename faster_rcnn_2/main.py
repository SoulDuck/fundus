#-*- coding:utf-8 -*-
import sys
sys.path.insert(0, '../')
from cnn import convolution2d , affine , dropout
from bbox_transform import bbox_transform_inv , clip_boxes
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
import random
import image_preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class FasterRcnnConv5():
    def __init__(self , n_classes  , eval_mode , data_dir):
        self.n_classes  = n_classes
        self.eval_mode = eval_mode
        self.data_dir = data_dir
        self._read_names()
        self.x_ = tf.placeholder(dtype=tf.float32 , shape = [1 , None , None ,1 ])
        self.im_dims = tf.placeholder(tf.int32 , [None ,2 ])
        self.gt_boxes = tf.placeholder(tf.int32 , [None ,5 ])
        self.phase_train = tf.placeholder(tf.bool)
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
        out_channels=[16, 16, 32, 64, 128]
        strides = [2, 1, 1, 2 ,1 ]
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
        RPN predicts the possibility of an anchor being background or foreground, and refine the anchor.
        """

        rpn_out_ch = 256
        rpn_k=3
        self.anchor_scales = [11, 13, 16]  # original anchor_scales
        n_anchors = len(self.anchor_scales) * 3 # len(ratio) =3
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
                self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights = \
                    anchor_target_layer.anchor_target_layer(rpn_cls_score=self.rpn_cls_layer, gt_boxes=self.gt_boxes,
                                                            im_dims=self.im_dims,
                                                            _feat_stride=self._feat_stride,
                                                            anchor_scales=self.anchor_scales)
                # layer shape : 1 ? ? 18
                # gt.boxes placeholder : ? ,5
                # img_dim : ? 2
            with tf.variable_scope('bbox'):
                # Bounding-Box regression layer (bounding box predictions)
                rpn_bbox_layer = convolution2d('conv', self.rpn_layer, out_ch=n_anchors * 4, k=1, s=1,act=None)  # (1, ?, ?, 36)
            self.rpn_bbox_layer = tf.identity(rpn_bbox_layer , 'bbox_output')
            print '** bbox_layer shape : {}'.format(np.shape(self.rpn_bbox_layer ))
            print

    def _roi_proposal(self):
        print '###### ROI Proposal Network building.... '
        print
        self.num_classes = 10 +1 # 1 -> background
        self.rpn_cls_prob=self._rpn_softmax()
        key = 'TRAIN' if self.eval_mode is False else 'TEST'
        self.blobs, self.scores = proposal_layer.proposal_layer(rpn_bbox_cls_prob=self.rpn_cls_prob,
                                                                rpn_bbox_pred=self.rpn_bbox_layer,
                                                                im_dims=self.im_dims, cfg_key=key,
                                                                _feat_stride=self._feat_stride,
                                                                anchor_scales=self.anchor_scales)
    
        if self.eval_mode is False:
            # Calculate targets for proposals
            self.rois, self.labels, self.bbox_targets, self.bbox_inside_weights, self.bbox_outside_weights = \
                proposal_target_layer.proposal_target_layer(rpn_rois=self.blobs, gt_boxes=self.gt_boxes,
                                      _num_classes=self.num_classes)
        else:
            # test
            self.rois=self.blobs
    def _fast_rcnn(self):
        print '###### Fast R-CNN building.... '
        print
        with tf.variable_scope('fast_rcnn'):
            keep_prob = cfg.FRCNN_DROPOUT_KEEP_RATE if self.eval_mode is False else 1.0
            self.pooledFeatures , self.boxes , self.box_ind = roi_pool.roi_pool(self.top_conv, self.rois, self.im_dims) #roi pooling
            layer = self.pooledFeatures # ? 7,7 128 Same Output
            # print layer
            for i in range(len(cfg.FRCNN_FC_HIDDEN)):
                layer = affine('fc_{}'.format(i), layer, cfg.FRCNN_FC_HIDDEN[i])
                layer = dropout(layer, phase_train=self.phase_train, keep_prob=keep_prob)
            with tf.variable_scope('cls'):
                self.fast_rcnn_cls_logits = affine('cls_logits', layer, self.num_classes, activation=None)
            with tf.variable_scope('bbox'):
                self.fast_rcnn_bbox_logits = affine('bbox_logits', layer, self.num_classes * 4, activation=None)

    def _optimizer(self):

        self.lr=0.01
        self.step=0
        # rpn optimzer

        self.rpn_cls_loss=loss_functions.rpn_cls_loss(self.rpn_cls_layer,self.rpn_labels)
        self.rpn_bbox_loss = loss_functions.rpn_bbox_loss(rpn_bbox_pred=self.rpn_bbox_layer,
                                                          rpn_bbox_targets=self.rpn_bbox_targets,
                                                          rpn_inside_weights=self.rpn_bbox_inside_weights,
                                                          rpn_outside_weights=self.rpn_bbox_outside_weights)


        # fast-rcnn optimzer
        self.fast_rcnn_cls_loss=loss_functions.fast_rcnn_cls_loss(self.fast_rcnn_cls_logits, self.labels)
        self.fast_rcnn_bbox_loss = loss_functions.fast_rcnn_bbox_loss(fast_rcnn_bbox_pred=self.fast_rcnn_bbox_logits,
                                                                      bbox_targets=self.bbox_targets,
                                                                      roi_inside_weights=self.bbox_inside_weights,
                                                                      roi_outside_weights=self.bbox_outside_weights)
        self.cost=tf.reduce_sum(self.rpn_cls_loss)# + self.rpn_bbox_loss)

                                #self.fast_rcnn_cls_loss + self.fast_rcnn_bbox_loss)#self.rpn_cls_loss
        #self.rpn_bbox_loss + self.fast_rcnn_cls_loss + self.fast_rcnn_bbox_loss
        """
        decay_steps = cfg.TRAIN.LEARNING_RATE_DECAY_RATE * len(self.train_names)  # Number of Epochs x images/epoch
        learning_rate = tf.train.exponential_decay(learning_rate=self.lr, global_step=self.step,
                                                   decay_steps=decay_steps, decay_rate=cfg.TRAIN.LEARNING_RATE_DECAY,
                                                   staircase=True)
        """
        learning_rate=0.01

        # Optimizer: ADAM
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)

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
        #tf_inputs = (self.x_, self.im_dims, self.gt_boxes)

        print self.step
        for self.epoch in trange(1, self.file_epoch + 1, desc='epochs'):
            for i in tqdm(train_order):
                feed_dict=self._create_feed_dict_for_train(i)
                try:
                    _, loss, fr_labels, fr_cls = self.sess.run(
                        [self.optimizer, self.cost, self.labels, self.fast_rcnn_cls_logits],
                        feed_dict=feed_dict)

                    proposal_rpn_bbox,proposal_rpn_scores , roi_pool_boxes, roi_pool_index = self.sess.run(
                        [self.blobs ,self.scores, self.boxes, self.box_ind], feed_dict=feed_dict)

                    #image 정보에 대한 tensor
                    rois,image_size,ori_img= self.sess.run([self.rois,self.im_dims ,self.x_],feed_dict=feed_dict)


                    #loss 정보에 대한 tensor
                    rpn_cls_loss, rpn_bbox_loss, fast_rcnn_cls_loss, fast_rcnn_bbox_loss = self.sess.run(
                        [self.rpn_cls_loss, self.rpn_bbox_loss, self.fast_rcnn_cls_loss, self.fast_rcnn_bbox_loss],
                        feed_dict=feed_dict)

                    #roi pooling에 대한 정보
                    rpn_cls, rpn_bbox, fr_cls, fr_bbox = self.sess.run(
                        [self.rpn_cls_prob, self.rpn_bbox_layer, self.fast_rcnn_cls_logits, self.fast_rcnn_bbox_logits],
                        feed_dict=feed_dict)

                    if self.step % 100 ==0:
                        print 'rpn_cls', rpn_cls[0, 0, 0, :10]
                        print 'rpn cls loss :', rpn_cls_loss
                        print 'rpn bbox loss :', rpn_bbox_loss
                        print 'fastr rcnn cls loss :', fast_rcnn_cls_loss
                        print 'fast rcnn bbox loss : ', fast_rcnn_bbox_loss
                        print 'rpn cls : ', np.shape(rpn_cls[0, :, :, :9])
                        self._save_proposal_rpn_bbox(ori_img , proposal_rpn_bbox , proposal_rpn_scores)


                    self.step+=1
                    """
                    print np.shape(rpn_cls)
                    print 'rpn_cls',rpn_cls[0,0,0,:10]
                    print 'roi pool indices'
                    print roi_pool_index
                    print 'rpn cls loss :',rpn_cls_loss
                    print 'rpn bbox loss :',rpn_bbox_loss
                    print 'fastr rcnn cls loss :',fast_rcnn_cls_loss
                    print 'fast rcnn bbox loss : ',fast_rcnn_bbox_loss

                    print 'rpn cls : ' , np.shape(rpn_cls[0,:,:,:9])
                    """

                    #self._show_result(rois,fr_cls , fr_bbox , image_size  ,ori_img )

                except Exception as e:
                    print e
                    exit()

    def _save_proposal_rpn_bbox(self , img , rpn_bbox , rpn_score , root_dir = './rpn_bbox'):
        def _make_folder():
            if not os.path.isdir(root_dir):
                os.makedirs(root_dir)
            count = 0;
            while(True):
                save_dir = os.path.join(root_dir, str(count))
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                    return save_dir
                else:
                    count+=1


        assert len(rpn_bbox) == len(rpn_score)
        fg_keep = [rpn_score > 0.5]
        fg_bboxes=rpn_bbox[fg_keep]
        save_dir=_make_folder()
        print 'save dir :',save_dir
        print 'saving....'
        if len(fg_bboxes) > 200:
            indices=random.sample(range(len(fg_bboxes)) ,200)
            fg_bboxes=fg_bboxes[indices]
        print 'fg_bboxes',len(fg_bboxes)
        for i,bbox in enumerate(fg_bboxes):
            fig = plt.figure()
            ax=fig.add_subplot(111)
            ax.imshow(img.reshape(img.shape[1:3]))
            x1,y1,x2,y2=bbox[1:]
            rect=patches.Rectangle((x1,y1) ,x2-x1 ,y2-y1 ,fill=False , edgecolor='w')
            ax.add_patch(rect)
            save_path = os.path.join(save_dir , str(i)+'.png' )
            plt.savefig(save_path)
            plt.close()


        print 'save proposal rpn bbox with image'


    def _show_result(self, rois , cls, bbox , im_dims , img):

        target_bbox=np.zeros([len(bbox) , 4])
        cls = np.argmax(cls, axis=1)

        for i,c in enumerate(cls):
            target_bbox[i,:]=bbox[i, c * 4:c * 4 + 4]
        pred_boxes=bbox_transform_inv(rois[:1] , target_bbox)
        pred_boxes = clip_boxes( pred_boxes,np.squeeze(im_dims))


        for i,c in enumerate(cls):
            if c !=0:
                fig = plt.figure()
                ax =fig.add_subplot(111)
                img=img.reshape(img.shape[1:3])
                ax.imshow(img)
                coord = pred_boxes[c]
                x1,y1,x2,y2=coord
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1 ,fill=False , edgecolor='w')
                ax.add_patch(rect)
                count=0
                print x1,y1,x2,y2
                print c
                #while (os.path.isfile('{}.png'.format(count))):
                #    count+=1
                plt.savefig('./{}.png'.format(c))
                plt.close()
                print c,'saved!'
                print np.shape(pred_boxes)

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
        feed_dict = {self.x_: img, self.im_dims: im_dims, self.gt_boxes : gt_bbox , self.phase_train : True}
        return feed_dict

    def _start_session(self):
        config = tf.ConfigProto(log_device_placement=False)
        rpn_saver = tf.train.Saver(max_to_keep=1)
        self.sess = tf.Session(config=config)
        init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
        self.sess.run(init)


if __name__ == '__main__':
    data_dir ='./clutteredMNIST'
    model = FasterRcnnConv5(n_classes=10, eval_mode=False, data_dir=data_dir)
    model.train(file_epoch=1)


