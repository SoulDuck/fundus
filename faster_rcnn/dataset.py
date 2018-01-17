#-*- coding:utf-8 -*-
import numpy as np
import xml.etree.ElementTree as ET
import os
import cv2
import tqdm

#loading Pascal Dataset



pascal_num_class = 21
pascal_class_names = {0: 'person', 1: 'bird', 2: 'cat', 3: 'cow', 4: 'dog', 5: 'horse', 6: 'sheep', 7: 'aeroplane', 8: 'bicycle', 9: 'boat', 10: 'bus', 11: 'car', 12: 'motorbike', 13: 'train', 14: 'bottle', 15: 'chair', 16: 'diningtable', 17: 'pottedplant', 18: 'sofa', 19: 'tvmonitor', 20: 'background'}
pascal_class_ids = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19, 'background': 20}
pascal_class_colors = [[174, 220, 192], [116, 108, 127], [118, 144, 153], [189, 149, 122], [191,  93, 101], [154, 190, 115], [216, 148, 110], [230, 141, 249], [191, 217, 206], [156, 111, 135], [138, 147, 168], [138, 241, 227], [171, 113, 234], [139, 208, 147], [123, 205, 243], [145, 116, 119], [206, 204, 195], [157, 174, 227], [194, 205, 238], [183, 184, 164], [152, 248, 224]]

class DataSet():
    def __init__(self, img_ids, img_files, img_heights, img_widths, batch_size=1, anchor_files=None, gt_classes=None,
                 gt_bboxes=None, is_train=False, shuffle=False):
        self.img_ids = np.array(img_ids)
        self.img_files = np.array(img_files)
        self.img_heights = np.array(img_heights)
        self.img_widths = np.array(img_widths)
        self.anchor_files = np.array(anchor_files)
        self.batch_size = batch_size
        self.gt_classes = gt_classes
        self.gt_bboxes = gt_bboxes
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.current_index = 0
        self.count = len(self.img_files)
        self.indices = list(range(self.count))
        self.num_batches = int(self.count / self.batch_size)
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()
        start, end = self.current_index, self.current_index + self.batch_size
        current_indices = self.indices[start:end]
        img_files = self.img_files[current_indices]
        if self.is_train:
            anchor_files = self.anchor_files[current_indices]
            self.current_index += self.batch_size
            return img_files, anchor_files
        else:
            self.current_index += self.batch_size
            return img_files

    def has_next_batch(self):
        """ Determine whether there is any batch left. """
        return self.current_index + self.batch_size <= self.count


def prepare_train_pascal_data(args):
    """ Prepare relevant PASCAL data for training the model. """
    image_dir, annotation_dir, data_dir = args.train_pascal_image_dir, args.train_pascal_annotation_dir, args.train_pascal_data_dir
    batch_size = args.batch_size
    basic_model = args.basic_model
    num_roi = args.num_roi
    files = os.listdir(annotation_dir)
    img_ids = list(range(len(files)))

    img_files = []
    img_heights = []
    img_widths = []
    anchor_files = []
    gt_classes = []
    gt_bboxes = []

    for f in files:
        try:
            annotation = os.path.join(annotation_dir, f)

            tree = ET.parse(annotation)
            root = tree.getroot()
            img_name = root.find('filename').text # filename 2007_00027.jpg
            img_file = os.path.join(image_dir, img_name)
            img_files.append(img_file) # 여기에 이미지 파일을 준다.

            img_id_str = os.path.splitext(img_name)[0] #2007_000027

            size = root.find('size')
            img_height = int(size.find('height').text)
            img_width = int(size.find('width').text)
            img_heights.append(img_height)
            img_widths.append(img_width)
            anchor_files.append(os.path.join(data_dir, img_id_str + '_' + basic_model + '_anchor.npz'))

            classes = []
            bboxes = []

            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = pascal_class_ids[class_name]
                classes.append(class_id)

                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                bboxes.append([ymin, xmin, ymax - ymin + 1, xmax - xmin + 1])

            gt_classes.append(classes)
            gt_bboxes.append(bboxes)
        except Exception as e:
            print 'Error filename : {} {} '.format(img_name, [bndbox.find('xmin').text, bndbox.find('ymin').text,
                                                              bndbox.find('xmax').text, bndbox.find('ymax').text, ])



    print("Building the training dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths, batch_size, anchor_files, gt_classes, gt_bboxes,
                      True, True)
    print("Dataset built.")
    return dataset


def prepare_val_pascal_data(args):
    """ Prepare relevant PASCAL data for validating the model. """
    image_dir, annotation_dir = args.val_pascal_image_dir, args.val_pascal_annotation_dir

    files = os.listdir(annotation_dir)
    img_ids = list(range(len(files)))

    img_files = []
    img_heights = []
    img_widths = []

    pascal = {}

    for f in files:
        annotation = os.path.join(annotation_dir, f)

        tree = ET.parse(annotation)
        root = tree.getroot()

        img_name = root.find('filename').text
        pascal[img_name] = []

        img_file = os.path.join(image_dir, img_name)
        img_files.append(img_file)

        size = root.find('size')
        img_height = int(size.find('height').text)
        img_width = int(size.find('width').text)
        img_heights.append(img_height)
        img_widths.append(img_width)

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = pascal_class_ids[class_name]
            temp = obj.find('difficult')
            difficult = int(temp.text) if temp != None else 0

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            pascal[img_name].append({'class_id': class_id, 'bbox': [xmin, ymin, xmax, ymax], 'difficult': difficult})

    print("Building the validation dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths)
    print("Dataset built.")
    return pascal, dataset


def eval_pascal_one_class(pascal, detections, c):
    """ Evaluate the detection result for one class on PASCAL dataset. """
    gts = {}
    num_objs = 0
    for img_name in pascal:
        gts[img_name] = []
        for obj in pascal[img_name]:
            if obj['class_id'] == c and obj['difficult'] == 0:
                gts[img_name] += [{'bbox': obj['bbox'], 'detected': False}]
                num_objs += 1

    dts = []
    scores = []
    num_dets = 0
    for img_name in detections:
        for dt in detections[img_name]:
            if dt['class_id'] == c:
                dts.append([img_name, dt['bbox'], dt['score']])
                scores.append(dt['score'])
                num_dets += 1

    # Sort the detections based on their scores
    scores = np.array(scores, np.float32)
    sorted_idx = np.argsort(scores)[::-1]

    tp = np.zeros((num_dets))
    fp = np.zeros((num_dets))

    for i in tqdm(list(range(num_dets))):
        idx = sorted_idx[i]
        img_name = dts[idx][0]
        bbox = dts[idx][1]
        gt_bboxes = np.array([obj['bbox'] for obj in gts[img_name]], np.float32)

        # Compute the max IoU of current detection with the ground truths
        max_iou = 0.0
        if gt_bboxes.size > 0:
            ixmin = np.maximum(gt_bboxes[:, 0], bbox[0])
            iymin = np.maximum(gt_bboxes[:, 1], bbox[1])
            ixmax = np.minimum(gt_bboxes[:, 2], bbox[2])
            iymax = np.minimum(gt_bboxes[:, 3], bbox[3])

            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)

            area_intersect = iw * ih
            area_union = (bbox[2] - bbox[0] + 1.0) * (bbox[3] - bbox[1] + 1.0) + (gt_bboxes[:, 2] - gt_bboxes[:,
                                                                                                    0] + 1.0) * (
                                                                                 gt_bboxes[:, 3] - gt_bboxes[:,
                                                                                                   1] + 1.0) - area_intersect

            ious = area_intersect / area_union
            max_iou = np.max(ious, axis=0)
            j = np.argmax(ious)

        # Determine if the current detection is a true or false positive
        if max_iou > 0.5:
            if not gts[img_name][j]['detected']:
                tp[i] = 1.0
                gts[img_name][j]['detected'] = True
            else:
                fp[i] = 1.0
        else:
            fp[i] = 1.0

    # Accumulate the numbers of true and false positives
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    # Compute the average precision based on these data
    rec = tp * 1.0 / num_objs
    prec = tp * 1.0 / np.maximum((tp + fp), np.finfo(np.float64).eps)

    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    print('average precision for class %s = %f' % (pascal_class_names[c], ap))

    return ap


def eval_pascal(pascal, detections):
    """ Evaluate the detection result on PASCAL dataset. """
    ap = 0.0
    for i in range(pascal_num_class - 1):
        ap += eval_pascal_one_class(pascal, detections, i)
    ap = ap / (pascal_num_class - 1)
    print('mean average precision = %f' % ap)
    return ap


def prepare_test_data(args):
    """ Prepare relevant data for testing the model. """
    image_dir = args.test_image_dir

    files = os.listdir(image_dir)
    files = [f for f in files if f.lower().endswith('.jpg')]

    img_ids = list(range(len(files)))
    img_files = []
    img_heights = []
    img_widths = []

    for f in files:
        img_path = os.path.join(image_dir, f)
        img_files.append(img_path)
        img = cv2.imread(img_path)
        img_heights.append(img.shape[0])
        img_widths.append(img.shape[1])

    print("Building the testing dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths)
    print("Dataset built.")
    return dataset



if '__main__' == __name__:



    annotation_dir ='/Users/seongjungkim/PycharmProjects/fundus/dataset/Annotations'
    files = os.listdir('/Users/seongjungkim/PycharmProjects/fundus/dataset/Annotations');
    img_ids = list(range(len(files))) #17125
    anchor_files = []
    img_heights = []
    img_widths = []
    anchor_files = [] # anchor files?
    gt_classes = [] #
    gt_bboxes = [] #

    for f in files:
        annotation = os.path.join(annotation_dir, f)
        tree = ET.parse(annotation)
        root = tree.getroot()
        img_name = root.find('filename').text
        img_id_str = os.path.splitext(img_name)[0]
        size = root.find('size')
        img_height = int(size.find('height').text)
        img_width = int(size.find('width').text)
        img_heights.append(img_height)
        img_widths.append(img_width)
        #anchor_files.append(os.path.join(data_dir, img_id_str + '_' + basic_model + '_anchor.npz')) #basic_model = 'vgg16' , #data_dir = 'pascal'
        classes = []
        bboxes = []
        for i,obj in enumerate(root.findall('object')):
            class_name = obj.find('name').text # e.g) person
            class_id = pascal_class_ids[class_name] # e.g) 0
            classes.append(class_id)
            bndbox = obj.find('bndbox')

            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bboxes.append([ymin, xmin, ymax - ymin + 1, xmax - xmin + 1])
            print xmin, xmax , ymax , ymax
            if i == 2:
                exit()
        print classes
        gt_classes.append(classes)
        gt_bboxes.append(bboxes)

        dataset = DataSet(img_ids, img_files, img_heights, img_widths, batch_size, anchor_files, gt_classes, gt_bboxes,
                          is_train=True, shuffle=True)
        #return dataset



            #print class_name

