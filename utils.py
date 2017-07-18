import sys,os
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
from PIL import Image
import random
"""
import Image

background = Image.open("bg.png")
overlay = Image.open("ol.jpg")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.5)
new_img.save("new.png","PNG")
"""
def get_name(path):
    name = path.split('/')[-1].split('.')[0]
    return name
def check_overlay_paths(all_paths , src_paths):
    """
    return not overlay image btw param all_paths and src_paths
    :param all_paths:
    :param src_paths:
    :return:
    """
    return_paths=[]
    overlay_paths=[]
    src_names=map(get_name,src_paths)
    for path in all_paths:
        name=path.split('/')[-1].split('.')[0]
        if name in src_names:
            overlay_paths.append(path)
        else:
            return_paths.append(path)
    if __debug__ ==True:
        print 'the number of overlay images : ',len(overlay_paths)
    return return_paths
def compare_images(ori_img, target_img):
    plt.title('debuging')
    fig= plt.figure()
    a=fig.add_subplot(1,2,1)
    a.set_xlabel('original image')
    plt.imshow()

    a = fig.add_subplot(1, 2, 1)
    plt.imshow('changed image')
    plt.show()
def show_progress(i,max_iter):
    msg='\r progress {}/{}'.format(i, max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()
def plot_images(imgs , names=None):
    h=math.ceil(math.sqrt(len(imgs)))
    fig=plt.figure()

    for i in range(len(imgs)):
        ax=fig.add_subplot(h,h,i+1)
        ind=random.randint(0,len(imgs)-1)
        img=imgs[ind]
        plt.imshow(img)
        if not names==None:
            ax.set_xlabel(names[ind])
    plt.savefig('./1.png')
    plt.show()
def open_images(paths):
    imgs=[]
    for path in paths:
        if path.endswith('.npy'):
            img=np.load(path)
        else:
            img=Image.open(path)
            img=np.asarray(img)
        imgs.append(imgs)
    return imgs
def sorted_fundus(paths):
    imgs=open_images(paths)
    n,h,w,c=np.shape(imgs)
    center=(h/2, w/2)
    for img in imgs:
        l_sum=img[:,:center[1],:].sum()
        r_sum=img[:,center[1]:,:].sum()
    if l_sum > r_sum:
        plt.imshow(img)
        plt.show()

def change_mode(image, mode='RGB'):
    #image = Image.fromarray(image * 255)
    image = image.convert(mode)
    return image

def np2img(image):
    try:
        image=Image.fromarray(image)
        return image
    except:
        print 'input value isnt numpy type '
        return Image

def delete_char_from_paths(folder_path , del_char):
    folder_names=os.walk(folder_path).next()[1]
    for folder_name in folder_names:
        paths=glob.glob(folder_path+folder_name+'/*.png')
        for path in paths:
            new_path=path.replace(del_char ,'')
            os.rename(path , new_path)

    """
    *usage:
        delete_char_from_paths(folder_path='../fundus_data/cropped_macula/' , del_char='*')
        test*.txt --> test.txt
    """
def get_paths_from_text(text_locate):
    f=open(text_locate , 'r')
    lines=f.readlines()
    lines=map(lambda x: x.replace('\n' , '' ) , lines)

    return lines





"""
def overlay_images(front_image , back_image):
    try:
        front_image=Image.fromarray(front_image)
    except:
        pass
    try:
        back_image=Image.fromarray(back_image)
    except:
        pass

    back_image.paste(front_image, (0, 0), front_image)
    back_image.show()
    return b_image
"""



def make_log_txt(folder_path):
    f = open(folder_path+'/log.txt', 'a')
    return f

def write_acc_loss(f,train_acc,train_loss,test_acc,test_loss):
    f.write(str(train_acc)+'\t'+str(train_loss)+'\t'+str(test_acc)+'\t'+str(test_loss)+'\n')

def divide_images_labels_from_batch(images, labels ,batch_size):
    batch_img_list=[]
    batch_lab_list = []
    share=len(labels)/batch_size
    #print len(images)
    #print len(labels)
    #print 'share :',share

    for i in range(share):

        imgs=images[i*batch_size:(i+1)*batch_size]
        labs=labels[i * batch_size:(i + 1) * batch_size]
       # print i , len(imgs) , len(labs)
        batch_img_list.append(imgs)
        batch_lab_list.append(labs)
        if i==share-1:
            imgs = images[-batch_size:]
            labs = labels[-batch_size:]
            #print i+1, len(imgs), len(labs)
            batch_img_list.append(imgs)
            batch_lab_list.append(labs)

    return batch_img_list , batch_lab_list
def plot_xs_ys(title,xs_title, ys_title , folder_path,xs ,*arg_ys ):
    plt.xlabel(xs_title)
    plt.ylabel(ys_title)
    plt.title(title)
    for ys in arg_ys:
        ys=list(ys)
        plt.plot(xs, ys)
        #folder_path = './graph/' + file_path.split('/')[-1].split('.')[0]
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    plt.savefig(folder_path +'/'+ys_title)
    plt.close()

def draw_grpah(file_pointer,check_point=50):
    if isinstance(file_pointer , str):
        file_path=file_pointer
        f=open(file_path,'r')
    else:
        f=file_pointer

    lines=f.readlines()
    train_acc_list=[];train_loss_list=[];val_acc_list=[];val_loss_list=[];step_list=[]
    for i,line in enumerate(lines):
        step=i*check_point
        step_list.append(step)
        train_acc, train_loss , val_acc , val_loss=line.split('\t')
        train_acc_list.append(train_acc);train_loss_list.append(train_loss);val_acc_list.append(val_acc);val_loss_list.append(val_loss)
    if __debug__==True:
        print 'the number of steps',len(step_list)
        print 'the number of train accuracy , loss',len(train_acc_list),len(train_loss_list)
        print 'the number of validation accuracy and loss',len(val_acc_list) , len(val_loss_list)
    folder_path = './graph/' + file_path.split('/')[-1].split('.')[0]
    plot_xs_ys('Normal Vs Abnormal','Step','Train Accuracy',folder_path,step_list , train_acc_list)
    plot_xs_ys('Normal Vs Abnormal', 'Step', 'Train Loss', folder_path,step_list, train_loss_list )
    plot_xs_ys('Normal Vs Abnormal', 'Step', 'Validation Accuracy', folder_path,step_list, val_acc_list)
    plot_xs_ys('Normal Vs Abnormal', 'Step', 'Validation Loss', folder_path,step_list, val_acc_list)
    plot_xs_ys('Normal Vs Abnormal','Step','Train_Validation Accuracy ',folder_path,step_list, train_acc_list, val_acc_list)
    plot_xs_ys('Normal Vs Abnormal','Step','Train_Validation Loss ',folder_path,step_list, train_loss_list, val_loss_list)
def make_folder(root_folder_path , folder_name):
    count=0
    w_flag=True
    while w_flag:
        if not os.path.isdir(root_folder_path+folder_name+str(count)):
             os.mkdir(root_folder_path+folder_name+str(count))
             w_flag=False
        else:
           count+=1
    return root_folder_path+folder_name+str(count)+'/'

if __name__=='__main__':
    #make_log_txt()
    #delete_char_from_paths(folder_path='../fundus_data/cropped_macula/', del_char='*')
    """
    paths=glob.glob('./sample_image/original_images/*.png')
    imgs=open_images(paths)
    print np.shape(imgs)
    plot_images(imgs)
    """
    """
    #function divide_images_labels_from_batch test
    test_imgs=np.load('./test_imgs.npy')
    test_labs=np.load('./test_labs.npy')
    imgs,labs=divide_images_labels_from_batch(test_imgs , test_labs ,batch_size=60)
    print labs[8]
    """
    #draw_grpah('./log/79%_top_of_batch_STEM_REDUCTION_A_B_ABNORMAL_NORMAL.txt')