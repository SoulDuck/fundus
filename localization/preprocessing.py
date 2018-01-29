#-*- coding:utf-8 -*-
import numpy as np
import utils
from fundus_processing import overlaps , get_width_height
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.patches as patches
def dense_crop(image , crop_height , crop_width , lr_flip =False, ud_flip=False ):
    """
     _________________
    | ____       ___  |
    ||    |-->->|   | |
    ||____|     |___| |

            ...
      ____       ____
    ||    |-->->|    ||
    ||____|     |____||
    |_________________|



    :param image:
    :param crop_height:
    :param crop_width:
    :param lr_flip:
    :param ud_flip:
    :return:
    """
    coords=[]
    cropped_images=[]
    img_h,img_w,ch=np.shape(image)
    n_h_move = img_h - crop_height + 1
    n_w_move = img_w - crop_width + 1
    for h in range(n_h_move):
        for w in range(n_w_move):
            x1 = w
            y1 = h
            x2 = w + crop_width
            y2 = h + crop_height
            coords.append((x1,y1,x2,y2))
            cropped_images.append(image[ h : h+crop_height , w : w+crop_width , :])
    assert len(cropped_images) == len(coords)
    return cropped_images , coords



class preprocessing(object):
    def __init__(self , csv_dir , img_dir):
        self.csv_dir  = csv_dir
        self.img_dir = img_dir

        # 전체 csv paths 을 training 과 test csv 로 나눈다
        self.csv_paths = glob.glob(os.path.join(self.csv_dir, '*.csv'))
        self.n_paths = len(self.csv_paths)
        self.n_test_paths = int(self.n_paths * 0.1)
        self.n_train_paths = self.n_paths - self.n_test_paths
        self.test_csv_paths = self.csv_paths[:self.n_test_paths]
        self.train_csv_paths = self.csv_paths[self.n_test_paths:]
        self.all_labels=self._get_all_coords() # get all_labels

        self._get_cropped()

    def get_coords(self, path):
        labels = {}
        f = open(path, 'r')
        lines = f.readlines()
        for i, line in enumerate(lines[1:]):
            label, x1, y1, x2, y2 = map(float, line.split(','))
            if not label in labels.keys():
                labels[int(label)] = [[x1, y1, x2, y2]]  #
            else:
                labels[int(label)].append([x1, y1, x2, y2])  # )

        return labels

    def _get_all_coords(self):
        assert len(self.csv_paths) > 0 , 'the number of csv path {} '.format(len(self.csv_paths))
        self.all_labels={}
        for path in self.train_csv_paths:
            labels=self.get_coords(path)
            for key in labels.keys():
                if not key in self.all_labels.keys():
                    self.all_labels[key] = labels[key]
                else:
                    self.all_labels[key].extend(labels[key])
        return self.all_labels

    def _get_cropped(self):
        root_root_roi_dir = './data/roi'
        root_root_fg_dir = './data/fg'
        root_root_bg_dir = './data/bg'


        if not os.path.exists(root_root_roi_dir):
            os.makedirs(root_root_roi_dir)
            print 'a'
        if not os.path.exists(root_root_fg_dir):
            os.makedirs(root_root_fg_dir)
        if not os.path.exists(root_root_bg_dir):
            os.makedirs(root_root_bg_dir)
        for path in self.train_csv_paths:
            name=os.path.split(path)[1]
            name = os.path.splitext(name)[0]
            labels = self.get_coords(path) # csv별 roi을 가져온다. 예시 [4] : [[x1,y1 x2, y2] ...[x1,y1 x2, y2]]

            # root_root_roi , fg_dir , bg_dir 에서 csv 이름이 있는 폴더를 생성한다
            # E.G) './data/roi'-->'./data/roi/140352'
            root_roi_dir, root_fg_dir, root_bg_dir = map(lambda path: os.path.join(path, name),
                                                         [root_root_roi_dir, root_root_fg_dir, root_root_bg_dir])
            if not os.path.isdir(root_roi_dir):
                map(lambda path: os.makedirs(os.path.join(path)),[root_roi_dir, root_fg_dir, root_bg_dir])
            # 폴더가 생성된 적이 없다면 폴더를 생성한다
            """
            data 이런식의 파일 경로를 가지고 있다
             |-bg
             |  |-1
             |  |-2
             |  ...
             |  |-6
             |-fg
             |-roi
            """
            print 'name:',name
            for k in labels.keys():
                if not k ==1 : # 1번 라벨에 대해서만 fg을 얻는다
                    continue
                # make dir
                roi_dir, fg_dir, bg_dir = map(lambda path: os.path.join(path, str(k)),
                                              [root_roi_dir, root_fg_dir, root_bg_dir])
                if not os.path.isdir(roi_dir):
                    map(lambda path: os.makedirs(os.path.join(path)), [roi_dir, fg_dir, bg_dir])
                # load image
                img = np.asarray(Image.open(os.path.join(self.img_dir, name + '.png')))
                # crop image
                cropped_images, cropped_coords = dense_crop(img, 75, 75)  # 100 , 100 으로 모든 이미지를 자른다

                # save background image and foreground image to each folder
                for fg_coord in labels[k]:

                    print 'foreground coord : {}'.format(fg_coord)
                    fg_x1, fg_y1, fg_x2, fg_y2=map(int , fg_coord)
                    fg_w = fg_x2 - fg_x1
                    fg_h = fg_y2 - fg_y1
                    fg_area = fg_w * fg_h

                    fig, ax = plt.subplots(1)
                    ax.imshow(img)
                    rect=patches.Rectangle((fg_x1, fg_y1), fg_w, fg_h , facecolor=None , linewidth=1 , edgecolor='r' ,fill=False)
                    ax.add_patch(rect)
                    plt.show()


                    for i,bg_coord in enumerate(cropped_coords):
                        utils.show_progress(i, len(cropped_coords))
                        bg_coord = map(int, bg_coord)
                        bg_x1, bg_y1, bg_x2, bg_y2 = bg_coord
                        area = overlaps(bg_coord , fg_coord)
                        if np.max(cropped_images[i]) <= 30:
                            continue
                        if area == None or area <=int(fg_area*0.3):
                            bg_img=Image.fromarray(cropped_images[i])
                            plt.imsave(os.path.join(bg_dir,str(i))+'.png' ,bg_img )
                        elif area >=int(fg_area*0.8): # foreground Image에 90퍼 이상 겹치면 foregound Image로 분류한다
                            #print area
                            fg_img = Image.fromarray(cropped_images[i])
                            plt.imsave(os.path.join(fg_dir, str(i)+'.png'), fg_img)

if __name__ =='__main__':
    img_dir ='/Users/seongjungkim/data/detection/resize'
    csv_dir='/Users/seongjungkim/data/detection/csv'
    model=preprocessing(csv_dir , img_dir)

    print len(model.all_labels[6])
    """
    for k in model.train_labels.keys():
        print len(model.train_labels[k])
    print model.train_labels[2]
    width_height=map(lambda coord : get_width_height(coord) , model.train_labels[1])

    count =0
    for w,h in width_height:
        if w < 100 and h < 100 :
            count +=1
            print w ,h
            plt.scatter(w,h)
        print count


    plt.show()
    exit()
    
    img=Image.open('../debug/0.png').convert('RGB')
    np_img=np.asarray(img)

    print np.shape(np_img)
    cropped_images , coords=dense_crop(np_img,30,30)

    for coord in coords:
        cropped_images = img.crop(coord)
        plt.imshow(cropped_images)
        plt.show()

    """
