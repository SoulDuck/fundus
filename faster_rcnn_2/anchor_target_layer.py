# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:11:17 2017
@author: Kevin Liang (modifications)
Anchor Target Layer: Creates all the anchors in the final convolutional feature
map, assigns anchors to ground truth boxes, and applies labels of "objectness"
Adapted from the official Faster R-CNN repo: 
https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/anchor_target_layer.py
"""

# --------------------------------------------------------
# Faster R-CNN
# Written by KimSeongJung
# --------------------------------------------------------
import sys
#sys.path.append('../')

import numpy as np
import numpy.random as npr
import tensorflow as tf

from configure import cfg
import bbox_overlaps
import bbox_transform
import generate_anchor


#py_func은 session 이 실행되고 나서 검사된다. 그 전에는 그냥 검사만 추가한다.
def anchor_target_layer(rpn_cls_score, gt_boxes, im_dims, _feat_stride, anchor_scales):
    '''
    Make Python version of _anchor_target_layer_py below Tensorflow compatible
    '''
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
        tf.py_func(_anchor_target_layer_py, [rpn_cls_score, gt_boxes, im_dims, _feat_stride, anchor_scales],
                   [tf.float32, tf.float32, tf.float32, tf.float32])

    rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
    rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')
    rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')
    rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights, name='rpn_bbox_outside_weights')

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

def _anchor_target_layer_py(rpn_cls_score, gt_boxes, im_dims, _feat_stride, anchor_scales):
    """
    Python version    
    이해가 안가는게 ... 가끔 겹치지 않는 sample들이 있는데 그게 불가능 한데...뭐지....
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap
    """
    im_dims = im_dims[0]
    # _anchors shape : ( 9, 4 ) anchor coordinate type : x1,y1,x2,y2
    _anchors = generate_anchor.generate_anchors(scales=np.array(anchor_scales))

    _num_anchors = _anchors.shape[0]
    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # Only minibatch of 1 supported
    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'
    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]
    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose() # 4,88 을 88,4 로 바꾼다
    A = _num_anchors # 9
    K = shifts.shape[0] # 88
    all_anchors=np.array([])

    for i in range(len(_anchors)):
        if i ==0 :
            all_anchors=np.add(shifts , _anchors[i])
        else:
            all_anchors = np.concatenate((all_anchors, np.add(shifts, _anchors[i])), axis=0)

    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_dims[1] + _allowed_border) &  # <-- width
        (all_anchors[:, 3] < im_dims[0] + _allowed_border))[0] # <-- height

    anchors = all_anchors[inds_inside]
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    overlaps = bbox_overlaps.bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)) #anchor 별로 얼마나 겹치는지 확인해준다

    argmax_overlaps = overlaps.argmax(axis=1) # 여러 gt box 중에 가장 많이 겹치는 gt 을 가져온다
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps] # inds_inside 갯수 만큼 overlaps에서 가장 높은 overlays
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    #print gt_argmax_overlaps # gt_argmax_overlap 이 empty가 뜨는데 어떻게 해결해야 하지.....
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])] #[ 0.63559322  0.39626705]

    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1 # 가장 높은 anchor의 라벨은 1로 준다

    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE) # fg 와 bg 을 1:1 로 맞추어야 한다 .
    fg_inds = np.where(labels == 1)[0]


    print 'the number of all lables : ', np.shape(all_anchors)
    print 'the number of inside labels : ', np.shape(anchors)
    print 'the number of positive labels :', np.sum(labels == 1), '(anchor_target_layer.py)'
    print 'positive overlaps : '
    print
    # print anchors[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP]
    """
    print gt_boxes
    for gt in gt_boxes:
        x1, y1, x2, y2, l = gt
        print x2 - x1, y2 - y1
    """

    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False) # replace = False --> 겹치지 않게 한다
        labels[disable_inds] = -1
    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
    # fg 는 무조건 하나 포함되는데 그 이유는 max IOU을 가지고 있는건 무조건 FG로 보게 한다
    # bg or fg 가 지정한 갯수보다 많으면 -1 라벨해서 선택되지 않게 한다
    # bbox_targets: The deltas (relative to anchors) that Faster R-CNN should 
    # try to predict at each anchor
    # TODO: This "weights" business might be deprecated. Requires investigation

    #bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32) 이게 왜 필요하지
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :]) #  bbox_targets = dx , dy , dw , dh
    #Regression 을 할수 있게 변형한다
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS) #(1.0, 1.0, 1.0, 1.0)

    # Give the positive RPN examples weight of p * 1 / {num positives}
    # and give negatives a weight of (1 - p)
    # Set to -1.0 to use uniform example weighting

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0: #TRAIN.RPN_POSITIVE_WEIGHT = -1
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0) # get positive label
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        #print 'positive weight ',positive_weights
        #print 'negative weight ',negative_weights
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights
    # map up to original set of anchors

    #ins_inside
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2) # A = 9
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    rpn_bbox_targets = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    # bbox_inside_weights
    rpn_bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    # bbox_outside_weights
    rpn_bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform.bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
