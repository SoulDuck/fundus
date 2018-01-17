import sys
from cnn import convolution2d
import os
import os.path as osp
import numpy as np
from distutils import spawn
import tensorflow as tf
from configure import cfg
import anchor_target_layer
class FasterRcnnConv5():
    def __init__(self , n_classes ):
        self.n_classes  = n_classes
        self.x_ = tf.placeholder(dtype=tf.float32 , shape = [1 , None , None ,1 ])
        self.im_dims = tf.placeholder(tf.int32 , [None ,2 ])
        self.gt_boxes = tf.placeholder(tf.int32 , [None ,5 ])
        self._build()

    def _build(self):
        self._convnet() #1.convolution 5 layer
        self._rpn()
        self._roi_proposal()
        #2.feature map
        #3.rpn
        #4.roi proposal network
        #5.rcn

    def _convnet(self):
        kernels=[5, 3, 3, 3, 3]
        out_channels=[32, 64, 64, 128, 128]
        strides = [2, 2, 1, 2, 1]
        layer=self.x_
        print np.prod(strides)

        for i in range(5):
            layer = convolution2d(name='conv_{}'.format(i), x=layer, out_ch=out_channels[i], k=kernels[i], s=strides[i],
                                  padding='SAME')
        self.top_conv = tf.identity(layer , 'top_conv')
        self._feat_stride = np.prod(strides)

    def _rpn(self):

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
                layer = convolution2d('conv' ,layer , out_ch= n_anchors*2 , k=1 , act=None)
        with tf.variable_scope('target'):
            """
            print layer
            print self.gt_boxes
            print self.im_dims
            print self._feat_stride
            print anchor_scales
                """
            self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights = anchor_target_layer.anchor_target_layer(
                rpn_cls_score=layer, gt_boxes=self.gt_boxes, im_dims=self.im_dims,
                _feat_stride=self._feat_stride, anchor_scales=anchor_scales)
            # layer shape : 1 ? ? 18
            # gt.boxes placeholder : ? ,5
            # img_dim : ? 2

            print self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights
            with tf.variable_scope('bbox'):
                # Bounding-Box regression layer (bounding box predictions)
                layer = convolution2d('conv', layer, out_ch=n_anchors * 4, k=1, act=None)

            self.rpn_output = tf.identity(layer, 'rpn_output')
            #'rpn fileter size ?' rpn output channel?

    def _roi_proposal(self):
        # proposal highest IOU boxes

if __name__ == '__main__':
    model=FasterRcnnConv5(10)
