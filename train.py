import warnings
import helpers
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


warnings.simplefilter(action='ignore', category=FutureWarning)

# model = ResNet50(weights=None,input_shape=(384, 384 ,3), classes=101)
model = VGG16(weights=None,input_shape=(64, 64, 3), classes=101)

# Compile classifier
# model.compile(loss='categorical_crossentropy',optimizer=optimizers.rmsprop(lr=0.0001, decay=1e-6))
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.00008, beta_1=0.9, beta_2=0.97, epsilon=1e-7))

# Fitting CNN to the images
train_data_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_data_generator = ImageDataGenerator(rescale=1./255)
training_set = train_data_generator.flow_from_directory('./food101/train', target_size=(128, 128), batch_size=4, class_mode='categorical')
test_set = test_data_generator.flow_from_directory('./food101/test', target_size=(128, 128), batch_size=64, class_mode='categorical')
model.fit_generator(training_set, steps_per_epoch=800/32, epochs=100, validation_data=test_set, validation_steps=200/32, shuffle=False)

helpers.save_model(model=model)
