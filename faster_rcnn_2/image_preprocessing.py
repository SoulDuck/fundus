from configure import cfg
import numpy as np

def _rearrange_channels(image):
    '''
    Flip RGB to BGR for pre-trained weights (OpenCV and Caffe are silly)
    Args:
        image (numpy array 3D)
    Returns:
        Rearranged image
    '''
    return image[:, :, (2, 1, 0)]

def _subtract_ImageNet_pixel_means(image):
    '''
    Subtract ImageNet pixel means found in config file
    Args:
        image (numpy array 3D)
    Returns:
        Demeaned image
    '''
    return image - cfg.PIXEL_MEANS

def image_preprocessing(image):
    '''
    Applies dataset-specific image pre-processing. Natural image processing
    (mean subtraction) done by default. Room to add custom preprocessing

    Args:
        image (numpy array 2D/3D): image to be processed
    Returns:
        Preprocessed image
    '''

    if cfg.NATURAL_IMAGE:
        image = _rearrange_channels(image)
        image = _subtract_ImageNet_pixel_means(image)

    ###########################################################################
    # Optional TODO: Add your own custom preprocessing for your dataset here
    ###########################################################################

    # Expand image to 4 dimensions (batch, height, width, channels)
    if len(image.shape) == 2:
        image = np.expand_dims(np.expand_dims(image, 0), 3)
    else:
        image = np.expand_dims(image, 0)

    return image