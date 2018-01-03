import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
from simplejson import loads

__C = edict()
cfg = __C
__C.TRAIN = edict()
__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.WEIGHT_DECAY = 0.0001
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.STEPSIZE = [30000] #Step size for reducing the learning rate, currently only support one step
__C.TRAIN.DISPLAY = 10
__C.TRAIN.DOUBLE_BIAS = True
__C.TRAIN.BIAS_DECAY = False
__C.TRAIN.USE_GT = False #GT --> Ground Truth
__C.TRAIN.ASPECT_GROUPING = False # -->  Whether to use aspect-ratio grouping of training images
__C.TRAIN.SNAPSHOT_KEPT = 3 # # the number of snapshot kept ,
__C.TRAIN.SUMMARY_INTERVAL = 180 # time interval for saving tensorflow summaries
__C.TRAIN.SCALES = (600,) #
__C.TRAIN.MAX_SIZE = 1000
__C.TRAIN.IMS_PER_BATCH = 1
__C.TRAIN.BATCH_SIZE = 128
__C.TRAIN.FG_FRACTION = 0.25 #Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_THRESH = 0.5 # Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)


# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1
__C.TRAIN.USE_FLIPPED = True
__C.TRAIN.BBOX_REG = True # bounding box regression
__C.TRAIN.BBOX_THRESH = 0.5
__C.TRAIN.SNAPSHOT_ITERS = 5000
__C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn'
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
__C.TRAIN.PROPOSAL_METHOD = 'gt'

__C.TRAIN.HAS_RPN = True #Region Proposal Network
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
__C.TRAIN.RPN_FG_FRACTION = 0.5
__C.TRAIN.RPN_BATCHSIZE = 256
__C.TRAIN.RPN_NMS_THRESH = 0.7
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
__C.TRAIN.USE_ALL_GT = True




if __name__ =='__main__':
    j = """{
    "Buffer": 12,
    "List1": [
        {"type" : "point", "coordinates" : [100.1,54.9] },
        {"type" : "point", "coordinates" : [109.4,65.1] },
        {"type" : "point", "coordinates" : [115.2,80.2] },
        {"type" : "point", "coordinates" : [150.9,97.8] }
    ]
    }"""
    d = edict(loads(j))
    print d.Buffer
    print d.List1[0]


    d = edict()
    d.foo = 3
    print d.foo
