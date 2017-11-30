import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from PIL import Image
import numpy as np
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

def binarization(np_img , threshold):
    assert np.ndim(np_img) ==2
    h,w=np.shape(np_img)
    flat_img=np.reshape(np_img , [-1,1])
    flat_img=np.squeeze(flat_img)
    n=len(flat_img)
    threshold=np.ones([n])*threshold
    np.squeeze(threshold)
    tmp=np.squeeze(np.int32([threshold <= flat_img])) # thr = 5 , img_ele =8  --> True
    flat_img=np.multiply(tmp, flat_img)
    res_img=np.reshape(flat_img , [h,w])

    plt.imshow(res_img)
    plt.show()
    plt.close()

    hist = cv2.calcHist([np_img], [0], None, [256], [0, 256])
    plt.hist(np_img.ravel(), 256, [0, 256])
    plt.title('Histogram for gray scale picture')
    plt.show()

#load Image
fundus_paths=glob.glob('./debug/*.png')

img=Image.open(fundus_paths[0])
img=Image.open('./debug/hard.png')
img=img.convert("RGB")
np_img=np.asarray(img)
h,w,ch=np.shape(np_img) # how to RGB?
print h,w,ch
#plt.imshow(img)
#plt.show()

# extract red channel from image
np_img=np.reshape(np_img , [1,h,w,ch])
red_np_img=np_img[:,:,:,0]
red_np_img=np.squeeze(red_np_img)
binarization(red_np_img , 210)
exit()
print 'red numpy image shape : {}'.format(np.shape(red_np_img))


#get gray image
gray_np_img=np.swapaxes(np_img , 1,3) # 1,300,300,3 --> 1,3,300,300
gray_np_img=rgb2gray(gray_np_img) # 1,1,300,300
#get histogram equalization image from gray image
histo_np_img=histo_equalized(gray_np_img)

gray_np_img=np.swapaxes(gray_np_img , 1,3) # 1,300,300,3 --> 1,300,300,1
gray_np_img=np.squeeze(gray_np_img) # 300,300

histo_np_img=np.swapaxes(histo_np_img , 1,3) # 1,300,300,3 --> 1,300,300,1
histo_np_img=np.squeeze(histo_np_img) # 300,300

plt.imshow(histo_np_img)
plt.show()



h_axes=int(h/6.)
w_axes=int(w/6.)
print h_axes , w_axes


np_img=np.swapaxes(np_img , 1,3)
np_img=np.squeeze(np_img)
#plt.imshow(np_img , cmap='Greys')
#plt.show()

def crop_by_grid( np_img ,cropImg_height ,cropImg_width   ):
    """
      _______ _______
     |<- w ->|<- w ->|
   | |       |       |
   h |       |       |
   | |_______|_______| .... <--image
     |       |       |
     |       |       |
     |       |       |
             .
             .
             .

    :param np_img:
    :param h:
    :param w:
    :return: cropped_images
    """

    img_h, img_w = np.shape(np_img)
    print img_h , img_w
    share_h = int(img_h/float(cropImg_height))
    share_w = int(img_w / float(cropImg_width))
    print share_h  , share_w
    cropped_images={}
    count=0

    max_sum , min_sum= 0,1000000
    for sh in range(share_h): # sh = share_height
        for sw in range(share_w): # wh = share_width
            print 'index h : {} , w : {}'.format(sh , sw)
            cropped_img=np_img[ cropImg_height*sh : cropImg_height*(sh+1) , cropImg_width*sw  :cropImg_width*(sw+1) ]
            pixel_sum=np.sum(cropped_img)
            if pixel_sum > max_sum :
                max_sum = pixel_sum
                cropped_images['max_sum'] = cropped_img
                print 'Max sum pixel Value : {}'.format(max_sum)
            if pixel_sum < min_sum :
                min_sum = pixel_sum
                cropped_images['min_sum'] = cropped_img
                print 'Min sum pixel Value : {}'.format(min_sum)
            cropped_images[count]=cropped_img
            print 'pixel sumValue  : ',pixel_sum
            #plt.imshow(cropped_img)
            #plt.show()
            #plt.close()
            count+=1
    return cropped_images

cropped_images=crop_by_grid(red_np_img ,cropImg_height=300 ,cropImg_width=150 )
cropped_images=crop_by_grid(cropped_images['max_sum'] ,cropImg_height=100 ,cropImg_width=150)
plt.imshow(cropped_images['max_sum'])
plt.show()
cropped_images=crop_by_grid(cropped_images['max_sum'] ,cropImg_height=100 ,cropImg_width=50)
plt.imshow(cropped_images['max_sum'])
plt.show()

plt.imshow(cropped_images[0])
plt.show()
plt.imshow(cropped_images[1])
plt.show()
plt.imshow(cropped_images[2])
plt.show()







# RGB IplImage to 4channel HSV image.
# Is there any way of converting 4channel RGB to HSV or 4channel RGB to 3channel RGB?
# RGBA

