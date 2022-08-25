import numpy as np
import matplotlib.pyplot as plt 
import random

# n = 10
# x = np.random.rand(n)
# y = np.random.rand(n)
# color_as_integer = np.random.randint(3, size=n)

# colormap = {
#     0  : np.array([  0,   0,   0, 255]),     # unlabelled
#     1  : np.array([ 70,  70,  70, 255]),     # building
#     2  : np.array([100,  40,  40, 255]),     # fence
# }

# # matplotlib works with rbga values in the range 0-1
# colormap = {k : v / 255. for k, v in colormap.items()}

# color_as_rgb = np.array([colormap[ii] for ii in color_as_integer])

# plt.scatter(x, y, s=100, c=color_as_rgb)
# # plt.show()
# plt.savefig('map.png') 

## wref: Nikhil using the below code
## https://colab.research.google.com/github/lexfridman/mit-deep-learning/blob/master/tutorial_driving_scene_segmentation/tutorial_driving_scene_segmentation.ipynb#scrollTo=vN0kU6NJ1Ye5

def create_label_colormap():
    """Creates a label colormap used in Cityscapes segmentation benchmark.

    Returns:
        A Colormap for visualizing segmentation results.
    """ 
    # manual colormap synthetic cityscapes
    # colormap = np.array([
    #     [26, 32, 30],
    #     [68, 74, 61],
    #     [89, 96, 79],
    #     [102, 102, 156],
    #     [100, 51, 43],
    #     [72, 82, 66],
    #     [250, 170, 30],
    #     [108, 116,  98],
    #     [50, 74, 59],
    #     [70, 74,  20],
    #     [50, 100, 160],
    #     [170,170,0],
    #     [70,70,0],
    #     [255,255,255],
    #     [10, 90,  90],
    #     [5, 50,  80],
    #     [10, 80, 90],
    #     [70, 70,  70],
    #     [200, 200,  200],
    #     [  0,   0,   0]], dtype=np.uint8)
    
    # manual2 colormap synthetic cityscapes
    colormap = np.array([
        [77, 77, 77],
        [100, 110, 120],
        [150,100,175],
        [102, 102, 156],
        [100, 51, 43],
        [125, 175, 175],
        [250, 170, 30],
        [220, 220,   0],
        [107, 142,  35],
        [70, 74,  20],
        [50, 100, 200],
        [190, 153, 153],
        [140,100,100],
        [255,255,255],
        [15, 25,  80],
        [50, 50,  190],
        [10, 80, 90],
        [180, 60,  50],
        [120, 50,  60],
        [  0,   0,   0]], dtype=np.uint8)  
    
    # colormap = np.array([
    #     [77, 87, 77],
    #     [78, 87, 75],
    #     [86, 95, 84],
    #     [70, 77, 66],
    #     [69, 77, 66],
    #     [67, 77, 69],
    #     [60, 69, 60],
    #     [80, 87, 80],
    #     [50, 61, 47],
    #     [60, 71, 53],
    #     [218, 231, 230],
    #     [64, 69, 62],
    #     [64, 69, 61],
    #     [62, 71, 65],
    #     [87, 94, 84],
    #     [70, 78, 70],
    #     [76, 91, 84],
    #     [59, 66, 58],
    #     [59, 66, 57],
    #     [  0,   0,   0]], dtype=np.uint8) # mean cityscapes val synthetic
    
    # colormap = np.array([
    #     [128,  64, 128],
    #     [244,  35, 232],
    #     [ 70,  70,  70],
    #     [102, 102, 156],
    #     [190, 153, 153],
    #     [153, 153, 153],
    #     [250, 170,  30],
    #     [220, 220,   0],
    #     [107, 142,  35],
    #     [152, 251, 152],
    #     [ 70, 130, 180],
    #     [220,  20,  60],
    #     [255,   0,   0],
    #     [  0,   0, 142],
    #     [  0,   0,  70],
    #     [  0,  60, 100],
    #     [  0,  80, 100],
    #     [  0,   0, 230],
    #     [119,  11,  32],
    #     [  0,   0,   0]], dtype=np.uint8) # cityscapes original label 
    
    # colormap = np.array([
    #     [69, 81, 72],
    #     [68, 78, 66],
    #     [68, 77, 66],
    #     [53, 59, 51],
    #     [54, 64, 51],
    #     [57, 67, 58],
    #     [44, 54, 45],
    #     [69, 77, 69],
    #     [38, 49, 35],
    #     [50, 63, 46],
    #     [229, 249, 247],
    #     [48, 54, 47],
    #     [50, 55, 46],
    #     [70, 79, 66],
    #     [46, 55, 48],
    #     [70, 79, 66],
    #     [50, 60, 53],
    #     [59, 76, 67],
    #     [46, 53, 45],
    #     [47, 56, 47]], dtype=np.uint8) # cityscapes median label
     
    return colormap

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

    colormap = create_label_colormap()
    # colormap = create_random_label_colormap() # random colormap 

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.') 

    return colormap[label]

LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle', 'void']) 

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1) 
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP) 


# unique_labels = np.unique(seg_map)
unique_labels = np.array([  0,   1,   2,   3,   4,   5,  6, 7,   8,   9,  10, 11,12, 13,14,15,16,17,18, 19],
      dtype=np.uint8)
plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
# ax.yaxis.tick_right()
plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
plt.xticks([], [])
# ax.tick_params(width=0.0)
plt.grid('off')
# plt.show() 
plt.savefig('city_manual_map.png')
