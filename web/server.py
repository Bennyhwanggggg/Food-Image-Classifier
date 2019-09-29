from flask import Flask, request, redirect, url_for, jsonify
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, Model, load_model
from PIL import Image
import requests
from werkzeug.utils import secure_filename


model_path = '../models/models_29_09_2019_15_57.h5'
weights_path = '../models/weights_29_09_2019_15_57.h5'

model = load_model(model_path)
model.load_weights(weights_path)

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify(message='error no file'), 400
    file = request.files['file']
    print(file)
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

    return jsonify(message='good'), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
