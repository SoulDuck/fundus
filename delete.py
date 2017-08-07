import os , glob
import numpy as np
from PIL import Image
import PIL

paths=glob.glob('/Users/seongjungkim/Downloads/FP300/*.jpg')
img=Image.open(paths[0])

h,w,ch=np.shape(img)
np_images=np.zeros([len(paths) , 300 ,300, ch])



f=open('./path.txt' , 'w')
for i,path in enumerate(paths):
    f.write(path+'\n')
    img=Image.open(path)
    img=img.resize([300,300] , PIL.Image.ANTIALIAS)
    img=np.asarray(img)
    np_images[i]=img


np.save('./FD_300.npy',np_images)
f.close()

images=np.load('./FD_300.npy')
print np.shape(images)
