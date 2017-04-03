import shutil
import glob
from random import random

def get_images_to_display(name):
    file_path = '{}*.jpeg'.format(name)
    for current_img_location in glob.glob(file_path):
        split = random()
        if split <= .7:
            new_image_location = 'data/train/' + name[5:]
        else:
            new_image_location = 'data/val/' + name[5:]
        shutil.copy(current_img_location, new_image_location)

# sort the images into class folders
zero = 'data/no_dr/'
one = 'data/mild/'
two = 'data/moderate/'
three = 'data/severe/'
four = 'data/proliferative/'

folders = [zero, one, two, three, four]
for fo in folders:
    get_images_to_display(fo)
