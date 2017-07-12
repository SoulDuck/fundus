import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import random
from PIL import ImageFilter
import aug

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
    imgs = map(aug.random_blur , imgs)
    imgs = map(aug.random_flip , imgs)
    imgs = map(aug.random_rotate, imgs)
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