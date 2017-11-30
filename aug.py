import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import random
from PIL import ImageFilter
import aug

def red_free_image(image):
    debug_flag = False
    # if not type(imgs).__module__ == np.__name__:
    try:
        if not type(image).__moduel__ == __name__:
            image=np.asarray(image)
    except AttributeError as attr_error:
        #print attr_error
        image = np.asarray(image)
    h,w,ch = np.shape(np.asarray(image))

    image_r = np.zeros([h,w])
    image_r.fill(0)
    image_g = image[:, :, 1]
    image_b = image[:, :, 2]

    image_r=image_r.reshape([h,w,1])
    image_g = image_g.reshape([h, w, 1])
    image_b = image_b.reshape([h, w, 1])


    image=np.concatenate([image_r , image_g, image_b] , axis=2)
    if __debug__ == debug_flag:
        print 'red_free_image debugging mode '
        print 'image red shape',np.shape(image_r)
        print 'red channel mean',image[:,:,0].mean()
        print 'image shape' , np.shape(np.asarray(image))
    return image



def green_free_image(image):
    # if not type(imgs).__module__ == np.__name__:
    try:
        if not type(image).__moduel__ == __name__:
            image=np.asarray(image)
    except AttributeError as attr_error:
        print attr_error
        image = np.asarray(image)
    h,w,ch = np.shape(np.asarray(image))

    image_r = image[:, :, 0]
    #image_g = image[:, :, 1]
    image_g = np.zeros([h,w])
    image_g.fill(0)
    image_b = image[:, :, 2]

    image_r=image_r.reshape([h,w,1])
    image_g = image_g.reshape([h, w, 1])
    image_b = image_b.reshape([h, w, 1])


    image=np.concatenate([image_r , image_g, image_b] , axis=2)
    if __debug__ == True:
        print 'image green shape',np.shape(image_g)
        print image[:,:,0].mean()
    return image


def blue_free_image(image):
    # if not type(imgs).__module__ == np.__name__:
    try:
        if not type(image).__moduel__ == __name__:
            image=np.asarray(image)
    except AttributeError as attr_error:
        print attr_error
        image = np.asarray(image)
    h,w,ch = np.shape(np.asarray(image))

    image_r = image[:, :, 0]
    image_g = image[:, :, 1]
    #image_b = image[:, :, 2]
    image_b = np.zeros([h,w])
    image_b.fill(0)

    image_r = image_r.reshape([h ,w ,1])
    image_g = image_g.reshape([h, w, 1])
    image_b = image_b.reshape([h, w, 1])


    image=np.concatenate([image_r , image_g, image_b] , axis=2)
    if __debug__ == True:
        print 'image blue shape',np.shape(image_b)
        print image[:,:,0].mean()
        print image[:, :, 1].mean()
        print image[:, :, 2].mean()
    return image

def check_type_numpy(a):
    if type(a).__module__ ==np.__name__:
        return True
    else:
        return False

def random_rotate(image):
    debug_flag=False
    if check_type_numpy(image):
        img=Image.fromarray(image)
    ### usage: map(random_rotate , images) ###
    ind=random.randint(0,180)
    minus = random.randint(0,1)
    minus=bool(minus)
    if minus==True:
        ind=ind*-1
    img=img.rotate(ind)
    img=np.asarray(img)
    #image type is must be PIL
    if __debug__ == debug_flag:
        print ind
    return img

def random_flip(image):
    debug_flag = False
    if not check_type_numpy(image):
        image=np.asarray(image)
    flipud_flag=bool(random.randint(0,1))
    fliplr_flag = bool(random.randint(0, 1))

    if flipud_flag== True:
        image=np.flipud(image)
    if fliplr_flag==True:
        image = np.fliplr(image)

    if __debug__==debug_flag:
        print 'flip lr ', str(fliplr_flag)
        print 'flip ud ', str(flipud_flag)
    return image
def random_blur(image):
    if check_type_numpy(image):
        image=Image.fromarray(image)
    ind=random.randint(0,10)
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=ind))
    blurred_image=np.asarray(blurred_image)

    return blurred_image
def aug_level_1(imgs):
    imgs = map(random_blur , imgs)
    imgs = map(random_flip , imgs)
    imgs = map(random_rotate, imgs)
    return imgs

def get_redfree_images(images):
    debug_flag= False
    if __debug__ ==debug_flag:
        print "get_redfree_images debug mode"
        print "image shape is ",np.shape(images)
    imgs = map(red_free_image, images)
    return imgs
if __name__ == '__main__':
    img=Image.open('./data/rion.png')
    img=random_rotate(img)
    img=random_flip(img)
    img=random_blur(img)
    #print np.shape(img)
    #img=img.rotate(45)
    #print np.shape(img)
    plt.imshow(img)
    plt.show()
    """
    img=cv2.imread('./data/rion.png',0)
    rows, cols=img.shape
    rotated_img=cv2.getRotationMatrix2D((cols/2, rows/2),90,1)
    img=np.asarray(img )
    img=img/255.
    print img.shape
    plt.imshow(img)
    plt.show()
    plt.imshow(rotated_img)
    plt.show()
    """

"""usage:red free image"""
"""
extension='png'
src_root_folder='../fundus_data/cropped_original_fundus_300x300/'
target_root_folder='../fundus_data/cropped_original_fundus_redfree/'
root_folder, sub_folder_names, file_list=os.walk(src_root_folder).next()
for sub_folder_name in sub_folder_names:
    src_folder=os.path.join(src_root_folder, sub_folder_name)
    saved_folder=os.path.join(target_root_folder , sub_folder_name)
    if not os.path.isdir(saved_folder):
        os.mkdir(saved_folder)
        print saved_folder+'is made'
    paths=glob.glob(src_folder +'/*.'+extension)
    images=map(Image.open , paths[:])
    names=map(lambda x : x.split('/')[-1].split('.')[0] ,paths[:3])
    start_time=time.time()
    redFree_images = map(red_free_image, images[:60])
    print np.shape(redFree_images)
    print  time.time() - start_time
"""