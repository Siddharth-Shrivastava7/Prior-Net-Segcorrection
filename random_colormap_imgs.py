import numpy as np
import matplotlib.pyplot as plt 
import random
import argparse 

parser = argparse.ArgumentParser() 

parser.add_argument('--num_perturb', type = int, default=7) 
args = parser.parse_args() 



def create_random_label_colormap():
    """Creates a label colormap used in Cityscapes segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """ 
    random_lst = [random.sample(range(1,256),3) for i in range(19)] 
    random_lst.append([0,0,0]) # invalid region
    random_colormap = np.array(random_lst, dtype=np.uint8) 
    return random_colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.
    Args:
        label: A 2D array with integer type, storing the segmentation label.
    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.
    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_random_label_colormap() # random colormap 

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.') 

    return colormap[label]

LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle', 'void']) 

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1) 
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP) 


unique_labels = np.array([  0,   1,   2,   3,   4,   5,  6, 7,   8,   9,  10, 11,12, 13,14,15,16,17,18, 19],
      dtype=np.uint8)
plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
plt.xticks([], [])
plt.grid('off')
plt.savefig('city_manual_map.png')
