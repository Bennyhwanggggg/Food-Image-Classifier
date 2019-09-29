import warnings
import numpy as np


warnings.simplefilter(action='ignore', category=FutureWarning)

import helpers
import h5py
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


# Config
batch_size = 64

# File Format
# f = h5py.File('./data/food_c101_n1000_r384x384x3.h5','r')
f = h5py.File('./data/food_c101_n10099_r64x64x3.h5', 'r')

x = np.array(f["images"])/255.
y = np.array([[int(i) for i in f["category"][j]] for j in range(len(f["category"]))])

# model = ResNet50(weights=None,input_shape=(384, 384 ,3), classes=101)
model = VGG16(weights=None,input_shape=(64, 64, 3), classes=101)
# model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))

# Compile classifier
# model.compile(loss='categorical_crossentropy',optimizer=optimizers.rmsprop(lr=0.0001, decay=1e-6))
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.97, epsilon=1e-7),
              metrics=['accuracy'])

train_x, test_x, train_y, test_y = train_test_split(x, y,test_size=0.2)

# Fitting CNN to the images
train_data_generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                          samplewise_center=False,  # set each sample mean to 0
                                          featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                          samplewise_std_normalization=False,  # divide each input by its std
                                          zca_whitening=False,  # apply ZCA whitening
                                          rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
                                          width_shift_range=0.125,  # randomly shift images horizontally (fraction of total width)
                                          height_shift_range=0.125,  # randomly shift images vertically (fraction of total height)
                                          horizontal_flip=True,  # randomly flip images
                                          vertical_flip=False, # randomly flip images
                                          rescale=1./255,
                                          fill_mode='nearest')
test_data_generator = ImageDataGenerator(rescale=1./255)
training_set = train_data_generator.flow(train_x,
                                         train_y,
                                         batch_size=batch_size)
test_set = test_data_generator.flow(test_x,
                                    test_y,
                                    batch_size=batch_size)
model.fit_generator(training_set,
                    steps_per_epoch=800/batch_size,
                    epochs=100,
                    validation_data=test_set,
                    validation_steps=200/batch_size,
                    shuffle=False)

helpers.save_model(model=model)
