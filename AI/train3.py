import os
import warnings


PATH = os.path.dirname(os.path.realpath(__file__))
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import h5py
import AI.helpers as helpers
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# File Format
# f = h5py.File('./data/food_c101_n1000_r384x384x3.h5', 'r')
f = h5py.File(os.path.join(PATH, './data/food_c101_n10099_r64x64x3.h5'), 'r')

print(f.keys())

# classifiers
# model = ResNet50(weights=None,input_shape=(64, 64, 3), classes=101)
model = VGG16(weights=None, input_shape=(64, 64, 3), classes=101)

x = np.array(f["images"])/255.
y = np.array([[int(i) for i in f["category"][j]] for j in range(len(f["category"]))])

print(len(f["category_names"]))
for cat in f["category"]:
    print(len(cat))
    break
sys.exit()

# classifier compile
model.compile(loss='categorical_crossentropy',optimizer=optimizers.rmsprop(lr=0.0001, decay=1e-6))
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.97, epsilon=1e-7),
#               metrics=["accuracy"])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
model.fit(train_x[:128], train_y[:128], batch_size=128, epochs=150, shuffle=False)


helpers.save_model(model=model)
