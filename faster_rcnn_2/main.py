import sys
from cnn import convolution2d , affine
import os
import os.path as osp
import numpy as np
from distutils import spawn
import tensorflow as tf
from configure import cfg
import anchor_target_layer
import proposal_layer
import proposal_target_layer
import roi_pool
class FasterRcnnConv5():
    def __init__(self , n_classes  , eval_mode):
        self.n_classes  = n_classes
        self.eval_mode = eval_mode
        self.x_ = tf.placeholder(dtype=tf.float32 , shape = [1 , None , None ,1 ])
        self.im_dims = tf.placeholder(tf.int32 , [None ,2 ])
        self.gt_boxes = tf.placeholder(tf.int32 , [None ,5 ])
        self._build()

    def _build(self):
        self._convnet() #1.convolution 5 layer
        self._rpn()
        self._roi_proposal()
        self._fast_rcnn()
        #2.feature map
        #3.rpn
        #4.roi proposal network
        #5.rcn

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
        anchor_scales = [8, 16, 32]
        n_anchors = len(anchor_scales) * 3 # len(ratio) =3
        #_n_anchors =len(self.anchor_scales)*3
        layer = self.top_conv
        with tf.variable_scope('rpn'):
                layer = convolution2d('conv', layer , out_ch= rpn_out_ch , k=rpn_k  , s=1 ,padding="SAME")
        with tf.variable_scope('cls'):
                cls_layer = convolution2d('conv' ,layer , out_ch= n_anchors*2 , k=1 , act=None)
                self.cls_layer = tf.identity(cls_layer, name='cls_output')
                print '** cls layer shape : {}'.format(np.shape(cls_layer)) #(1, ?, ?, 18)
        with tf.variable_scope('target'):
            """
            print layer
            print self.gt_boxes
            print self.im_dims
            print self._feat_stride
            print anchor_scales
            """
            self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights = anchor_target_layer.anchor_target_layer(
                rpn_cls_score=cls_layer, gt_boxes=self.gt_boxes, im_dims=self.im_dims,
                _feat_stride=self._feat_stride, anchor_scales=anchor_scales)
            # layer shape : 1 ? ? 18
            # gt.boxes placeholder : ? ,5
            # img_dim : ? 2
            with tf.variable_scope('bbox'):
                # Bounding-Box regression layer (bounding box predictions)
                bbox_layer = convolution2d('conv', layer, out_ch=n_anchors * 4, k=1, act=None) #(1, ?, ?, 36)
            self.bbox_layer = tf.identity(bbox_layer, 'bbox_output')
            print '** bbox_layer shape : {}'.format(np.shape(bbox_layer))
            #'rpn fileter size ?' rpn output channel?
            print
    def _roi_proposal(self):
        print '###### ROI Proposal Network building.... '
        print
        self.num_classes = cfg.NUM_CLASSES #10
        self.anchor_scales = cfg.RPN_ANCHOR_SCALES # [8, 16, 32]
        self.cls_prob=self._rpn_softmax()
        key = 'TRAIN' if self.eval_mode is False else 'TEST'

        self.blobs = proposal_layer.proposal_layer(rpn_bbox_cls_prob=self.cls_prob, rpn_bbox_pred=self.bbox_layer,
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
            pooledFeatures = roi_pool.roi_pool(self.top_conv, self.rois, self.im_dims)
            layer = pooledFeatures
            for i in range(len(cfg.FRCNN_FC_HIDDEN)):
                layer = affine('fc_{}'.format(i) , layer ,cfg.FRCNN_FC_HIDDEN[i])


            with tf.variable_scope('cls'):

            with tf.variable_scope('bbox'):

    def _rpn_softmax(self):

        shape=tf.shape(self.cls_layer)
        rpn_cls_score = tf.transpose(self.cls_layer,[0,3,1,2])
        rpn_cls_score = tf.reshape(rpn_cls_score,[shape[0],2,shape[3]//2*shape[1],shape[2]])
        rpn_cls_score = tf.transpose(rpn_cls_score,[0,2,3,1])

        rpn_cls_prob = tf.nn.softmax(rpn_cls_score)

        # Reshape back to the original
        rpn_cls_prob = tf.transpose(rpn_cls_prob, [0, 3, 1, 2])
        rpn_cls_prob = tf.reshape(rpn_cls_prob, [shape[0], shape[3], shape[1], shape[2]])
        rpn_cls_prob = tf.transpose(rpn_cls_prob, [0, 2, 3, 1])


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


if __name__ == '__main__':
    model=FasterRcnnConv5(10 , eval_mode=False)


