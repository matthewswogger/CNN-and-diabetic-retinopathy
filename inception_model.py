from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K

# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = 'data/train'
validation_data_dir = 'data/val'
batch_size = 16
num_classes = 5

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=180,
                                   shear_range=0.2,
                                   zoom_range=0.4,
                                   horizontal_flip=True,
                                   vertical_flip=True)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size)

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = val_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size)

# train model with a checkpointer to save the weights after every epoch
# checkpointer = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_acc:.2f}.hdf5')

model.fit_generator(train_generator,
                    steps_per_epoch=50,
                    epochs=2,
                    validation_data=validation_generator,
                    validation_steps=30,
                    verbose=1)#,
                    # callbacks=[checkpointer])
