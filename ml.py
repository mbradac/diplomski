# only the best model is here, rest of the models are available on
# https://colab.research.google.com/drive/1rWg1YFChR9kGp284rnSqmAfMTHP_nlMg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
        Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout)
import logging
import glob
import os
import json
import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array

random.seed(123)

# Setup logger.
logging.basicConfig(level='INFO')
logger = logging.getLogger('ml')


class Chaos2PoolFaceImage:
    NAME = "chaos2poolfaceimage"

    @staticmethod
    def build_model():
        def image_layers():
            eye_input = Input(shape=(64, 64, 3))
            conv1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(eye_input)
            max_pooling1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            conv2 = Conv2D(64, kernel_size=(3, 3),
                    activation="relu")(max_pooling1)
            max_pooling2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            flatten = Flatten()(max_pooling2)
            dropout = Dropout(0.3)(flatten)
            dense = Dense(128, activation="relu")(dropout)
            return eye_input, dense

        left_eye_input, left_eye_output = image_layers()
        right_eye_input, right_eye_output = image_layers()
        face_input, face_output = image_layers()

        all_data = keras.layers.concatenate(
                [left_eye_output, right_eye_output, face_output])
        dropout4 = Dropout(0.4)(all_data)
        all_data_dense2 = Dense(64, activation="relu")(dropout4)
        dropout5 = Dropout(0.4)(all_data_dense2)
        output = Dense(9, activation="softmax")(dropout5)

        model = Model(inputs=[left_eye_input, right_eye_input, face_input],
                outputs=[output])

        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['accuracy', 'mae', 'mse'])

        return model


def load_9point_dataset(input_name_prefixes, shuffle=True):
    input_names = []
    for input_name_prefix in input_name_prefixes:
        t = glob.glob(input_name_prefix + '*')
        t = filter(lambda x: os.path.splitext(x)[1] != '.jpg', t)
        t = list(map(lambda x: os.path.splitext(x)[0], t))
        input_names += t

    if shuffle:
        random.shuffle(input_names)

    face_points = []
    left_eyes = []
    right_eyes = []
    faces = []
    ys = []
    for i, input_name in enumerate(input_names):
        if i % 500 == 0:
            logger.info("Loading image {}/{}".format(i, len(input_names)))
        with open(input_name + '.json') as fp:
            data = json.load(fp)
        height = data["height"]
        width = data["width"]
        y_vector = [0] * 9
        y_vector[data["label"]] = 1
        ys.append(y_vector)
        points = []
        for landmark in sorted(data["landmarks"]):
            for x, y in data["landmarks"][landmark]:
                points.append((x, y))
        face_points.append(points)
        left_eyes.append(img_to_array(load_img(input_name + '_l.jpg')))
        right_eyes.append(img_to_array(load_img(input_name + '_r.jpg')))
        faces.append(img_to_array(load_img(input_name + '_f.jpg')))

    return (np.array(face_points), np.array(left_eyes) / 255.,
            np.array(right_eyes) / 255., np.array(faces) / 255., np.array(ys))


def train_9points_dataset(
        input_name_prefixes, test_input_name_prefixes):
    face_points, left_eyes, right_eyes, faces, ys = \
            load_9point_dataset(input_name_prefix)

    if test_input_name_prefix is None:
        size = len(face_points)
        train_size = int(0.8 * size)
        face_points_train, face_points_test = \
                face_points[:train_size], face_points[train_size:]
        left_eyes_train, left_eyes_test = \
                left_eyes[:train_size], left_eyes[train_size:]
        right_eyes_train, right_eyes_test = \
                right_eyes[:train_size], right_eyes[train_size:]
        faces_train, faces_test = \
                faces[:train_size], faces[train_size:]
        ys_train, ys_test = ys[:train_size], ys[train_size:]
    else:
        face_points_train = face_points
        left_eyes_train = left_eyes
        right_eyes_train = right_eyes
        faces_train = faces
        ys_train = ys
        face_points_test, left_eyes_test, right_eyes_test, faces_test, ys_test = \
                load_9point_dataset(test_input_name_prefix)

    model = Chaos2PoolFaceImage.build_model()
    model.summary()

    model.fit([left_eyes_train, right_eyes_train, faces_train],
            [ys_train],
            batch_size=128,
            epochs=25, # 12
            verbose=1,
            validation_data=(
                [left_eyes_test, right_eyes_test, faces_test], [ys_test]))

    if test_input_name_prefix is None:
        trained_model_path_template = "checkpoints/{}_{}__split_v2_weights"
    else:
        trained_model_path_template = "checkpoints/{}_{}__all_v2_weights"
    trained_model_path  = trained_model_path_template.format(
            Chaos2PoolFaceImage.NAME, "-".join(
                map(lambda x: os.path.basename(x), input_name_prefix)))
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_full_path = os.path.join(script_dir, trained_model_path)
    model.save_weights(model_full_path)

    return model


def evaluate_9points_dataset(model_path, test_input_name_prefixes):
    model = Chaos2PoolFaceImage.build_model()
    model.load_weights(model_path)
    logger.info("Model loaded")

    face_points, left_eyes, right_eyes, faces, ys = \
            load_9point_dataset(test_input_name_prefixes)
    print(model.evaluate([left_eyes, right_eyes, faces], ys))
