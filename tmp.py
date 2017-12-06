import argparse
import fundus_processing
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument('--nargs_int_type', nargs='+', type=int)
args=parser.parse_args()

print args.nargs_int_type
img_paths=glob.glob('./debug/*.png')
imgs = map(lambda img_path : Image.open(img_path).convert('RGB') , img_paths)
np_imgs = map(lambda img : np.asarray(img) , imgs)
print np.shape(np_imgs[0])
img , path =fundus_processing.crop_resize_fundus(path='/Users/seongjungkim/PycharmProjects/fundus/tmp/1_L.jpg')
plt.imshow(imgs[0])
plt.show()
print np.shape(imgs)


