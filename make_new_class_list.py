import pandas as pd
import glob

# get class labels for images
labels_df = pd.read_csv('trainLabels.csv')

# process to filter labels dataframe down to just the images I currently have
all_labels_dict = labels_df.set_index('image')['level'].to_dict()
labels_dict = {}
for image_path in glob.glob('data/*.jpeg'):
    image_name = image_path[5:-5]
    if image_name in all_labels_dict:
        labels_dict[image_name] = all_labels_dict[image_name]

# dataframe with just images I currently have
labels_df = pd.DataFrame(labels_dict.items(), columns=['image_name', 'class'])

# labels_df.to_csv('data/labels.csv')
