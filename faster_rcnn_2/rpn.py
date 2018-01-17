#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from cnn import *
class rpn(object):
    def __init__(self, top_conv, gt_boxes, im_dims, _feat_stride, eval_mode , anchor_scale):
        """

        :param self:
        :param conv_feat: need top conv
        :param gt_boxes:
        :param im_dims:
        :param _feat_stride:
        :param eval_mode:
        :return:
        """
        self.top_conv = top_conv
        self.gt_boxes = gt_boxes
        self.im_dims = im_dims
        self._feat_stride = _feat_stride
        self.anchor_scales = [8, 16, 32]
        self.anchor_ratio = []
        self.eval_mode = eval_mode
        self._build()


    def _build(self):
        with tf.varaible('rpn'):
            n_anchor = len(self.anchor_scale)*3  # n anchor has 3 ratio , 3 anchor

            layer = self.top_conv
            layer = convolution2d(name='rpn', x=layer, out_ch=512, k=1, padding='SAME')
            print layer
            print exit()
            with tf.variable('cls'):
                cls_layer=self.rpn_bbox_cls_layers.conv2d(filter_size=1, output_channels=n_anchor * 2, activation_fn=None)
                # 9개 roi boxes 에 대한 2 개의 classification 이 나온다.
                #cls_layer shape
            with tf.variable_scope('target'):
                # Only calculate targets in train mode. No ground truth boxes in evaluation mode
                if self.eval_mode is False:
                    # Anchor Target Layer (anchors and deltas)yhn
                    self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights = \
                        anchor_target_layer(rpn_cls_score=cls_layer, gt_boxes=self.gt_boxes, im_dims=self.im_dims,
                                            _feat_stride=self._feat_stride, anchor_scales=self.anchor_scales)

            with tf.variable_scope('bbox'):
                # Bounding-Box regression layer (bounding box predictions)
                self.rpn_bbox_pred_layers = Layers(features)
                self.rpn_bbox_pred_layers.conv2d(filter_size=1, output_channels=_num_anchors*4, activation_fn=None)






rpnrpn()

