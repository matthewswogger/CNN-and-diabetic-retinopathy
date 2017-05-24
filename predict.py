import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('th')

def cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     input_shape=(3, img_width, img_height), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                                                   metrics=['accuracy'])
    return model

# dimensions of our images.
img_width, img_height = 100, 100
test_data_dir = 'data/test'
batch_size = 16
num_classes = 5

# build the model
model = cnn_model()

# load model weights
model.load_weights('weights.52-0.76.hdf5')

# this is the augmentation configuration we will use for testing:
img = load_img('image_to_predict.jpg',False,target_size=(img_width,img_height))



# predict
predicted = model.predict(test_generator)
print(predicted)
