from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange
from scipy.ndimage.interpolation import zoom
from scipy.misc import imsave

import argparse
import numpy as np
import os
import shutil
import tensorflow as tf

# Global Flag Dictionary
flags = {
    'data_directory': './clutteredMNIST/',
    'nums': {"train": 55000, "valid": 5000, "test": 10000},
    'all_names': ["train", "valid", "test"],
    'num_classes': 10,
    'num_digits': 'random',  # Can also be a int in [1,2,3]. 'random' chooses an int [1, 3]
    'im_dims': 'random',  # Can also be an int > 28 * 2. 'random' chooses an int between [100, 200]
    # for each side
}

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='./Clutter MNIST Arguments')
    parser.add_argument('-t', '--train', default="PNG")
    parser.add_argument('-v', '--eval', default="PNG")
    args = vars(parser.parse_args())
    # Load and Convert Data
    all_data, all_labels = load_data()
    #shutil.rmtree("MNIST_data")

    # Create data directory
    make_directory(flags['data_directory'])

    # Create and save the cluttered MNIST digits
    process_digits(all_data, all_labels, flags['data_directory'], args)



def load_data():
    """ Download MNIST data from TensorFlow package, """
    mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

    train_data = mnist.train.images
    test_data = mnist.test.images
    valid_data = mnist.validation.images

    train_label = mnist.train.labels
    test_label = mnist.test.labels
    valid_label = mnist.validation.labels

    all_data = [train_data, valid_data, test_data]
    all_labels = [train_label, valid_label, test_label]
    return all_data, all_labels


def process_digits(all_data, all_labels, data_directory, args):
    """ Generate data and saves in the appropriate format """

    for s in range(len(flags['all_names'])):
        split = flags['all_names'][s]

        print('Processing {0} Data'.format(split))
        key = 'train' if split == 'train' else 'eval'

        # Create writer (tf_records) or Image/Annotations/Names directories (PNGs)
        if args[key] == 'tfrecords':
            tf_writer = tf.python_io.TFRecordWriter(data_directory + 'clutteredMNIST_' + split + '.tfrecords')
        elif args[key] == 'PNG':
            make_Im_An_Na_directories(data_directory)

        else:
            raise ValueError('{0} is not a valid data format option'.format(args[key]))

        # Generate data
        #print flags['nums'][split] --> 55000
        for i in trange(flags['nums'][split]):
            # Generate cluttered MNIST image
            im_dims = [np.random.randint(100, 200), np.random.randint(100, 200)] # make image size randomly
            num_digits =  np.random.randint(1, 3)
            img, gt_boxes = gen_nCluttered(all_data[s], all_labels[s], im_dims, num_digits)
            # Save data
            if args[key] == 'tfrecords':
                img = np.float32(img.flatten()).tostring()
                gt_boxes = np.int32(np.array(gt_boxes).flatten()).tostring()
                tf_write(img, gt_boxes, [flags['im_dims'], flags['im_dims']], tf_writer)
            elif args[key] == 'PNG':
                fname = split + '_img' + str(i)
                imsave(data_directory + 'Images/' + fname + '.png', np.float32(img))
                np.savetxt(data_directory + 'Annotations/' + fname + '.txt', np.array(gt_boxes), fmt='%i')
                with open(data_directory + 'Names/' + split + '.txt', 'a') as f:
                    f.write(fname + '\n')




def make_directory(folder_path):
    """Creates directory if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def make_Im_An_Na_directories(data_directory):
    '''Creates the Images-Annotations-Names directories for a data split'''
    make_directory(data_directory + 'Images/')
    make_directory(data_directory + 'Annotations/')
    make_directory(data_directory + 'Names/')

def im_dims_generator():
    """ Allow user to specify hardcoded image dimension or random rect dims """
    if flags['im_dims'] == 'random':
        return np.random.randint(100, 200)
    else:
        assert flags['im_dims'] > 0
        return flags['im_dims']
def num_digits_generator():
    """ Allow user to specify hardcoded number of digits or random num of digits """
    if flags['num_digits'] == 'random':
        return np.random.randint(1, 3)  # Can't do more than 3 digits, or the script will get stuck in the while loop
    else:
        assert flags['num_digits'] in [1, 2, 3]
        return flags['num_digits']


def gen_nCluttered(data, labels, im_dims, num_digits, clutter_rate=1):
    """
    Creates a clutterd MNIST image with a variable number of digits
    Args:
        data: MNIST image data (num_images, 784)
        labels: MNIST labels (num_images, )
        im_dims: [height, width] of output cluttered MNIST digit image
        num_digits: number of full MNIST digits to embed
        clutter_rate: how much to clutter background with fragments
    """
    # Initialize Blank image_out
    image_out = np.zeros([im_dims[0], im_dims[1]])
    max_val = 0
    gt_boxes = list()

    for i in range(num_digits):
        # Choose digit

        idx = np.random.randint(len(labels)) #len(labels) 55000
        digit = data[idx, :].reshape((28, 28))
        label = labels[idx]

        # Randomly Scale image
        h = np.random.randint(low=int(28 / 1.5), high=int(28 * 1.5))
        w = np.random.randint(low=int(28 / 1.5), high=int(28 * 1.5))
        digit = zoom(digit, (h / 28., w / 28.))
        count = 0;
        while True:
            # Randomly choose location in image_out
            x = np.random.randint(low=0, high=im_dims[1] - w)
            y = np.random.randint(low=0, high=im_dims[0] - h)
            print "x : {} , y : {}  ", x ,y
            # Ensure that digit doesn't overlap with another
            if np.sum(image_out[y:y + h, x:x + w]) == 0:
                break
            count += 1;
        # Insert digit into blank full size image and get max
        embedded = np.zeros([im_dims[0], im_dims[1]])
        print np.shape(embedded)
        embedded[y:y + h, x:x + w] += digit
        max_val = max(embedded.max(), max_val)
        # Assemble bounding box
        gt_bbox = create_gt_bbox(embedded, 12, label)

        # Save digit insertion
        image_out += embedded
        gt_boxes.append(gt_bbox)

    # Add in clutter patches of 8x8 dimension
    for j in range(int(np.mean(im_dims) * clutter_rate)):
        # Randomly select MNIST digit
        idx = np.random.randint(len(labels))
        fragment = data[idx, :].reshape(28, 28)

        # Randomly select patch of selected digit
        px = np.random.randint(low=0, high=28 - 8)
        py = np.random.randint(low=0, high=28 - 8)

        # Randomly choose location to insert clutter
        x = np.random.randint(low=0, high=im_dims[1] - 8)
        y = np.random.randint(low=0, high=im_dims[0] - 8)

        # Insert digit fragment, but not on top of digits
        if np.sum(image_out[y:(y + 8), x:(x + 8)]) == 0:
            image_out[y:(y + 8), x:(x + 8)] += fragment[py:(py + 8), px:(px + 8)]

    # Clip any over-saturated pixels
    image_out = np.clip(image_out, 0, max_val)

    # Subtract mean from image and scale to be between -1 and 1
    image_out -= image_out.mean()
    image_out = image_out / np.abs(image_out).max()

    return image_out, gt_boxes




if __name__ == '__main__':
    main()