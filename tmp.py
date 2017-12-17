import data
import aug
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import fundus_processing
import utils
train_imgs, train_labs, train_filenames, test_imgs, test_labs, test_filenames = data.type2('./fundus_300_debug' , save_dir_name='tmp')
train_imgs = train_imgs/255.
test_imgs = test_imgs /255.
print np.shape(train_imgs)
n_classes = 2

#rotated_imgs=map(lambda batch_x : aug.random_rotate(batch_x) , train_imgs[:60])
aug.random_rotate_image(train_imgs[:60])

""" image resize and crop test"""

img=test_imgs[0:1]
tf_ori_img=tf.Variable(img)
tf_img=tf.image.resize_image_with_crop_or_pad(tf_ori_img , 224, 224)
tf_resize_img=tf.image.resize_images(tf_ori_img , (224, 224))
"""brightness delta change"""
bright_delta_0=tf.image.adjust_brightness(img,delta=0)
bright_delta_1=tf.image.adjust_brightness(img,delta=0.05)
bright_delta_2=tf.image.adjust_brightness(img,delta=0.10)
bright_delta_3=tf.image.adjust_brightness(img,delta=0.15)
bright_delta_4=tf.image.adjust_brightness(img,delta=0.2)
#delta_random = tf.image.random_brightness(img, max_delta=0.2)

"""hue delta change"""
img=img.reshape((299,299,3))
hue_minus_delta_00 = tf.image.adjust_hue(img, delta=-0.01)
hue_minus_delta_01 = tf.image.adjust_hue(img, delta=-0.02)
hue_minus_delta_02 = tf.image.adjust_hue(img, delta=-0.03)
hue_minus_delta_03 = tf.image.adjust_hue(img, delta=-0.04)
hue_minus_delta_04 = tf.image.adjust_hue(img, delta=-0.05)
hue_minus_delta_05 = tf.image.adjust_hue(img, delta=-0.06)
hue_minus_delta_06 = tf.image.adjust_hue(img, delta=-0.07)
hue_minus_delta_07 = tf.image.adjust_hue(img, delta=-0.08)
hue_minus_delta_08 = tf.image.adjust_hue(img, delta=-0.09)

hue_minus_delta_0 = tf.image.adjust_hue(img, delta=-0.1)
hue_minus_delta_1 = tf.image.adjust_hue(img, delta=-0.2)
hue_minus_delta_2 = tf.image.adjust_hue(img, delta=-0.3)
hue_minus_delta_3 = tf.image.adjust_hue(img, delta=-0.4)
hue_minus_delta_4 = tf.image.adjust_hue(img, delta=-0.5)
hue_minus_delta_5 = tf.image.adjust_hue(img, delta=-0.6)
hue_minus_delta_6 = tf.image.adjust_hue(img, delta=-0.7)
hue_minus_delta_7 = tf.image.adjust_hue(img, delta=-0.8)
hue_minus_delta_8 = tf.image.adjust_hue(img, delta=-0.9)
hue_minus_delta_9 = tf.image.adjust_hue(img, delta=-1)

hue_delta_00 = tf.image.adjust_hue(img, delta=0.01)
hue_delta_01 = tf.image.adjust_hue(img, delta=0.02)
hue_delta_02 = tf.image.adjust_hue(img, delta=0.03)
hue_delta_03 = tf.image.adjust_hue(img, delta=0.04)
hue_delta_04 = tf.image.adjust_hue(img, delta=0.05)
hue_delta_05 = tf.image.adjust_hue(img, delta=0.06)
hue_delta_06 = tf.image.adjust_hue(img, delta=0.07)
hue_delta_07 = tf.image.adjust_hue(img, delta=0.08)
hue_delta_08 = tf.image.adjust_hue(img, delta=0.09)

hue_delta_0 = tf.image.adjust_hue(img, delta=0.1)
hue_delta_1 = tf.image.adjust_hue(img, delta=0.2)
hue_delta_2 = tf.image.adjust_hue(img, delta=0.3)
hue_delta_3 = tf.image.adjust_hue(img, delta=0.4)
hue_delta_4 = tf.image.adjust_hue(img, delta=0.5)
hue_delta_5 = tf.image.adjust_hue(img, delta=0.6)
hue_delta_6 = tf.image.adjust_hue(img, delta=0.7)
hue_delta_7 = tf.image.adjust_hue(img, delta=0.8)
hue_delta_8 = tf.image.adjust_hue(img, delta=0.9)
hue_delta_9 = tf.image.adjust_hue(img, delta=1)
"""delta change"""

"""contrast  delta change"""
contrast_imgs=[]
for i in range(-1000 , 1000 , 100 ):
    contrast_img = tf.image.adjust_contrast(img , contrast_factor=i/100.)
    contrast_img = tf.minimum(contrast_img, 1.0)
    contrast_img = tf.maximum(contrast_img, 0.0)
    contrast_imgs.append(contrast_img)

"""saturation delta change"""
saturation_imgs=[]
for i in range(-1000 , 1000 , 100 ):
    saturation_img = tf.image.adjust_saturation(img ,  saturation_factor=i/1000.)

    saturation_imgs.append(saturation_img)

#tf.image.adjust_contrast()
#tf.image.adjust_gamma()
#tf.image.adjust_saturation()
#tf.image.adjust_hue()


"""
 make Session
"""
sess= tf.Session()
init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)
output_img=sess.run(tf_img)
output_resize_img=sess.run(tf_resize_img)
output_img=np.squeeze(output_img)
output_resize_img=np.squeeze(output_resize_img)

"""
bright

"""
output_imgs=sess.run([bright_delta_0 ,bright_delta_1 ,bright_delta_2 ,bright_delta_3,bright_delta_4])
output_imgs=np.squeeze(output_imgs)
print np.max(output_imgs[0])
print np.max(output_imgs[1])
print np.max(output_imgs[2])
print np.max(output_imgs[3])
print np.max(output_imgs[4])
utils.plot_images(output_imgs)
"""
hue

"""
output_imgs_0=sess.run([hue_delta_00 ,hue_delta_01 ,hue_delta_02 ,hue_delta_03,hue_delta_04,\
                      hue_delta_05,hue_delta_06,hue_delta_07,hue_delta_08])

output_imgs_1=sess.run([hue_delta_0 ,hue_delta_1 ,hue_delta_2 ,hue_delta_3,hue_delta_4,\
                      hue_delta_5,hue_delta_6,hue_delta_7,hue_delta_8,hue_delta_9])

minus_output_imgs_0=sess.run([hue_minus_delta_00 ,hue_minus_delta_01 ,hue_minus_delta_02 ,hue_minus_delta_03,hue_minus_delta_04,\
                      hue_minus_delta_05,hue_minus_delta_06,hue_minus_delta_07,hue_minus_delta_08] )

minus_output_imgs_1=sess.run([hue_minus_delta_0 ,hue_minus_delta_1 ,hue_minus_delta_2 ,hue_minus_delta_3,hue_minus_delta_4,\
                      hue_minus_delta_5,hue_minus_delta_6,hue_minus_delta_7,hue_minus_delta_8,hue_minus_delta_9] )


output_imgs=np.squeeze(output_imgs_0)
output_imgs=np.squeeze(output_imgs_1)
minus_output_imgs=np.squeeze(minus_output_imgs_0)
minus_output_imgs=np.squeeze(minus_output_imgs_1)


#utils.plot_images(output_imgs_0)
#utils.plot_images(output_imgs_1)
#utils.plot_images(minus_output_imgs_0)
#utils.plot_images(minus_output_imgs_1)
"""
constrast 

"""

contrast_imgs=sess.run(contrast_imgs)
print np.shape(contrast_imgs)
utils.plot_images(contrast_imgs[:100])


"""
saturation 
"""
saturation_imgs=sess.run(saturation_imgs)
print np.shape(saturation_imgs)
utils.plot_images(saturation_imgs[:100])


exit()
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



"""augmentaiton check """

"""
 image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
"""



#tf.image.
