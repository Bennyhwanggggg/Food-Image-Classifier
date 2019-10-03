import warnings
import os

PATH = os.path.dirname(os.path.realpath(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


warnings.simplefilter(action='ignore', category=FutureWarning)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from keras.models import load_model
from keras.preprocessing import image as image_utils
from io import BytesIO
from PIL import Image
import tensorflow as tf
from file_manager import FileManager
from werkzeug.utils import secure_filename


# model files setup
file_manager = FileManager()

# local
# model_path = '../models/models_29_09_2019_18_32.h5'
# weights_path = '../models/weights_29_09_2019_18_32.h5'
# model_path = '../models/models_29_09_2019_18_50.h5'
# model_path = os.path.join(PATH, model_path)
# weights_path = '../models/weights_29_09_2019_18_50.h5'
# weights_path = os.path.join(PATH, weights_path)

# S3
model_file_to_download = 'models_03_10_2019_23_45.h5'
weights_file_to_download = 'weights_03_10_2019_23_45.h5'
model_file_name = 'models.h5'
weights_file_name = 'weights.h5'
model_path = os.path.join(PATH, model_file_name)
weights_path = os.path.join(PATH, weights_file_name)

if not os.path.exists(model_path):
    file_manager.download_file(model_file_to_download, model_path)
if not os.path.exists(weights_path):
    file_manager.download_file(weights_file_to_download, weights_path)

labels_file_path = os.path.join(PATH, 'labels.txt')
labels = np.loadtxt(labels_file_path, delimiter='\n', dtype=str)

model = load_model(model_path)
model.load_weights(weights_path)

if os.path.exists(model_path):
    os.remove(model_path)
if os.path.exists(weights_path):
    os.remove(weights_path)


global graph
graph = tf.get_default_graph()

app = Flask(__name__)
CORS(app)

@app.after_request
def after_request(resp):
    resp.headers["Access-Control-Allow-Origin"] = '*'
    request_headers = request.headers.get("Access-Control-Request-Headers")
    resp.headers["Access-Control-Allow-Headers"] = request_headers
    resp.headers['Access-Control-Allow-Methods'] = "DELETE, GET, POST, HEAD, OPTIONS"
    return resp


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify(message='error no file'), 400
    file = request.files['file']
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
    test_image = Image.open(BytesIO(data))
    test_image = test_image.resize((64, 64))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    with graph.as_default():
        result = model.predict(test_image, batch_size=1, verbose=1)
    for idx, n in enumerate(result[0]):
        if n == 1:
            print(idx)
    top_preds = result[0].argsort()[-5:][::-1]
    top_pred_names = [labels[i] for i in top_preds]
    print(top_pred_names)
    return jsonify(message=top_pred_names[0]), 200


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
