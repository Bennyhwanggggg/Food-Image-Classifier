import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)

from flask import Flask, request, redirect, url_for, jsonify
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, Model, load_model
from keras.preprocessing import image as image_utils
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from werkzeug.utils import secure_filename


# model_path = '../models/models_29_09_2019_18_32.h5'
# weights_path = '../models/weights_29_09_2019_18_32.h5'
model_path = '../models/models_29_09_2019_18_50.h5'
weights_path = '../models/weights_29_09_2019_18_50.h5'

labels = np.loadtxt('./labels.txt', delimiter='\n', dtype=str)

model = load_model(model_path)
model.load_weights(weights_path)


global graph
graph = tf.get_default_graph()

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify(message='error no file'), 400
    file = request.files['file']
    print(type(file))
    if not file.filename:
        return jsonify(message='error no file name'), 400
    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
        # file_path = os.path.join(TEMP_PATH, "{}-{}.txt".format(filename.rstrip('.txt'), int(time.time())))
        # file.save(file_path)
        # if not management_module.train(file_path):
        #     os.remove(file_path)
        #     return jsonify(message=UploadFileError.INVALID_FORMAT.value), 400
        # return jsonify(message=UploadFileSuccess.SUCCESS.value), 200
    data = file.read()
    print(type(data))
    test_image = Image.open(BytesIO(data))
    test_image = test_image.resize((64, 64))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    print(test_image)
    print(test_image.shape)
    with graph.as_default():
        result = model.predict(test_image, batch_size=1, verbose=1)
    # result = model.predict(test_image)
    print(result)
    for idx, n in enumerate(result[0]):
        if n == 1:
            print(idx)
    top_preds = result[0].argsort()[-5:][::-1]
    print(top_preds)
    top_pred_names = [labels[i] for i in top_preds]
    print(top_pred_names)
    return jsonify(message=top_pred_names[0]), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
