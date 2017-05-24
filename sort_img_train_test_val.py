from scipy import misc
import pandas as pd
import numpy as np

df = pd.read_csv('data/original_trainLabels.csv')

for f in ['a','b']:
    for name, l in zip(df.image, df.level):
        try:
            current_img_location = 'data/{0}/{1}.jpeg'.format(f, name)
            img = misc.imread(current_img_location)

            # figure out what class the image belongs to
            if l == 0:
                level = 'no_dr'
            elif l == 1:
                level = 'mild'
            elif l == 2:
                level = 'moderate'
            elif l == 3:
                level = 'severe'
            else:
                level = 'proliferative'

            # split data into train, test, val
            split = np.random.rand()
            if split <= 0.6:
                new_img_location = 'data/train/{0}/{1}.jpeg'.format(level, name)
            elif split <= 0.8:
                new_img_location = 'data/val/{0}/{1}.jpeg'.format(level, name)
            else:
                new_img_location = 'data/test/{0}/{1}.jpeg'.format(level, name)
            misc.imsave(new_img_location, img)
        except:
            pass
