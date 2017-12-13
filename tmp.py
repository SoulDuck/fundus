#-*- coding:utf-8 -*-
import argparse
import fundus_processing
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import json
import os
#### argparse : how to parse list
parser = argparse.ArgumentParser()
#parser.add_argument('--nargs_int_type', nargs='+', type=int)
#args=parser.parse_args()

"""
print args.nargs_int_type
img_paths=glob.glob('./debug/*.png')
imgs = map(lambda img_path : Image.open(img_path).convert('RGB') , img_paths)
np_imgs = map(lambda img : np.asarray(img) , imgs)
print np.shape(np_imgs[0])
img , path =fundus_processing.crop_resize_fundus(path='/Users/seongjungkim/PycharmProjects/fundus/tmp/1_L.jpg')
plt.imshow(imgs[0])
plt.show()
print np.shape(imgs)
"""

#### argparse : how to parse list

def _fn(step , lr_iters , lr_values):
    n_lr_iters = len(lr_iters)
    for idx in range(n_lr_iters):
        if step < lr_iters[idx]:
            return lr_iters[idx] , lr_values[idx]
        elif idx <= n_lr_iters - 1:
            continue
    return lr_iters[idx] , lr_values[idx]


step=range(100,2000 )
lr_iters=[200,40000,80000]
lr_values=[0.01,0.001,0.0001]
for s in step:
    iter_ , lr_value = _fn(s , lr_iters , lr_values)
    print '{} : {} : {}'.format(s , iter_ , lr_value)


for i in range(3):
    print i

os.path.join('asdf', None)