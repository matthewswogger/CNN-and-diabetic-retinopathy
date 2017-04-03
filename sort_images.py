import pandas as pd
import shutil

# sort the images into class folders
zero = 'data/no_dr/'
one = 'data/mild/'
two = 'data/moderate/'
three = 'data/severe/'
four = 'data/proliferative/'

for row in labels_df.itertuples():
    img_name = row[1]
    img_class = row[2]
    if img_class == 0:
        shutil.copy('data/' + img_name + '.jpeg', zero)
    elif img_class == 1:
        shutil.copy('data/' + img_name + '.jpeg', one)
    elif img_class == 2:
        shutil.copy('data/' + img_name + '.jpeg', two)
    elif img_class == 3:
        shutil.copy('data/' + img_name + '.jpeg', three)
    elif img_class == 4:
        shutil.copy('data/' + img_name + '.jpeg', four)
