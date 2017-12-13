import data
import aug
import numpy as np
train_imgs, train_labs, train_filenames, test_imgs, test_labs, test_filenames = data.type2('./fundus_300_debug' , save_dir_name='tmp')
train_imgs = train_imgs/255.
test_imgs = test_imgs /255.
print np.shape(train_imgs)
n_classes = 2

rotated_imgs=map(lambda batch_x : aug.random_rotate(batch_x) , train_imgs[:60])
