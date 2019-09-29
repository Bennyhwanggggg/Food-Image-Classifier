import os
import datetime

target_dir = '../models/'


def save_model(model, filename='{}.h5'.format(datetime.datetime.now().strftime("%d_%m_%Y_%H_%M"))):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    model_file = '{}models_{}'.format(target_dir, filename)
    weights_file = '{}weights_{}'.format(target_dir, filename)
    model.save(model_file)
    model.save_weights(weights_file)