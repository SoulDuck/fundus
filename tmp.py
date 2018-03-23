import numpy as np
import os , glob , shutil
preds=np.load('./iruda_preds.npy')
f=open('./iruda_paths.txt' ,'r')
lines=f.readlines()
image_dir ='/Users/seongjungkim/Desktop/iruda'
normal_dir='/Users/seongjungkim/Desktop/iruda_normal'
abnormal_dir='/Users/seongjungkim/Desktop/iruda_abnormal'

img_paths=glob.glob(image_dir+'*.JPG')
names=map(lambda path : os.path.splitext(os.path.split(path)) , img_paths)


preds_cls=np.argmax(preds , axis=1)


f_abnormal=open(os.path.join(abnormal_dir , 'names_preds'),'w')
f_normal=open(os.path.join(normal_dir , 'names_preds'),'w')
for i,line in enumerate(lines):
    name=os.path.splitext(os.path.split(line.replace('\n',''))[1])[0]
    path = os.path.join(image_dir, name + '.JPG')

    if preds_cls[i]== 0:  # abnormal
        shutil.copy( path ,os.path.join(abnormal_dir , name + '.JPG'))
        msg='{} : {}\n'.format(name , preds[i])
        f_abnormal.write(msg)
    elif preds_cls[i] == 1:  # normal
        shutil.copy(path,os.path.join(normal_dir, name + '.JPG'))
        msg='{} : {}\n'.format(name , preds[i])
        f_normal.write(msg)

    else:
        raise AssertionError






