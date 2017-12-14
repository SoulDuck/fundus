import data
import aug
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import fundus_processing
train_imgs, train_labs, train_filenames, test_imgs, test_labs, test_filenames = data.type2('./fundus_300_debug' , save_dir_name='tmp')
train_imgs = train_imgs/255.
test_imgs = test_imgs /255.
print np.shape(train_imgs)
n_classes = 2

#rotated_imgs=map(lambda batch_x : aug.random_rotate(batch_x) , train_imgs[:60])
aug.random_rotate_image(train_imgs[:60])

""" image resize and crop test"""

img=train_imgs[0:1]
tf_ori_img=tf.Variable(img)
tf_img=tf.image.resize_image_with_crop_or_pad(tf_ori_img , 224, 224)
tf_resize_img=tf.image.resize_images(tf_ori_img , (224, 224))
sess= tf.Session()
init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)
output_img=sess.run(tf_img)
output_resize_img=sess.run(tf_resize_img)
output_img=np.squeeze(output_img)
output_resize_img=np.squeeze(output_resize_img)

print np.shape(output_img)
print np.shape(output_resize_img)
plt.imshow(output_img)
plt.show()
plt.imshow(output_resize_img)
plt.show()


print np.shape(output_img)
print np.shape(output_img.reshape([1,224,224,3]))
images=fundus_processing.add_padding(output_img.reshape([1,224,224,3]) , 299,299)
images=np.squeeze(images)
plt.imshow(images)
plt.show()




