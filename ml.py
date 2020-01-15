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


class Chaos2Pool:
  NAME = "chaos2pool"

  @staticmethod
  def build_model():
      def eye_processing_layers():
          eye_input = Input(shape=(64, 64, 3))
          conv1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(eye_input)
          max_pooling1 = MaxPooling2D(pool_size=(2, 2))(conv1)
          conv2 = Conv2D(64, kernel_size=(3, 3), activation="relu")(max_pooling1)
          max_pooling2 = MaxPooling2D(pool_size=(2, 2))(conv2)
          flatten = Flatten()(max_pooling2)
          dropout = Dropout(0.5)(flatten)
          dense = Dense(128, activation="relu")(dropout)
          return eye_input, dense

      left_eye_input, left_eye_output = eye_processing_layers()
      right_eye_input, right_eye_output = eye_processing_layers()

      face_points_input = Input(shape=(72, 2))
      face_points_flatten = Flatten()(face_points_input)
      dropout1 = Dropout(0.5)(face_points_flatten)
      face_points_dense1 = Dense(64, activation="relu")(dropout1)
      dropout2 = Dropout(0.5)(face_points_dense1)
      face_points_dense2 = Dense(32, activation="relu")(dropout2)

      all_data = keras.layers.concatenate(
              [face_points_dense2, left_eye_output, right_eye_output])
      dropout3 = Dropout(0.5)(all_data)
      all_data_dense1 = Dense(64, activation="relu")(dropout3)
      dropout4 = Dropout(0.5)(all_data_dense1)
      all_data_dense2 = Dense(32, activation="relu")(dropout4)
      dropout5 = Dropout(0.5)(all_data_dense2)
      output = Dense(9, activation="softmax")(dropout5)

      model = Model(inputs=[left_eye_input, right_eye_input, face_points_input],
              outputs=[output])

      optimizer = tf.keras.optimizers.RMSprop(0.001)
      model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['accuracy', 'mae', 'mse'])

      return model


def load_9point_dataset(input_name_prefix, shuffle=True):
  input_names = glob.glob(input_name_prefix + '*')
  input_names = filter(lambda x: os.path.splitext(x)[1] != '.jpg', input_names)
  input_names = list(map(lambda x: os.path.splitext(x)[0], input_names))
  if shuffle:
    random.shuffle(input_names)

  face_points = []
  left_eyes = []
  right_eyes = []
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
                points.append((x / float(width), y / float(height)))
        face_points.append(points)
    with open(input_name + '_l.jpg') as fp:
        left_eyes.append(img_to_array(load_img(input_name + '_l.jpg')))
        right_eyes.append(img_to_array(load_img(input_name + '_r.jpg')))

  return (np.array(face_points), np.array(left_eyes) / 255.,
          np.array(right_eyes) / 255., np.array(ys))


def train_9points_dataset(modelclass, input_name_prefix):
  face_points, left_eyes, right_eyes, ys = load_9point_dataset(input_name_prefix)
  size = len(face_points)
  train_size = int(0.8 * size)

  face_points_train, face_points_test = \
          face_points[:train_size], face_points[train_size:]
  left_eyes_train, left_eyes_test = \
          left_eyes[:train_size], left_eyes[train_size:]
  right_eyes_train, right_eyes_test = \
          right_eyes[:train_size], right_eyes[train_size:]
  ys_train, ys_test = ys[:train_size], ys[train_size:]

  model = modelclass.build_model()
  model.summary()

  model.fit([left_eyes_train, right_eyes_train, face_points_train],
          [ys_train],
          batch_size=32,
          epochs=25, # 12
          verbose=1,
          validation_data=(
              [left_eyes_test, right_eyes_test, face_points_test], [ys_test]))

  trained_model_path  = "checkpoints/{}_{}__v0_weights".format(
      modelclass.NAME, os.path.basename(input_name_prefix))
  model.save_weights(trained_model_path)
  return model


def evaluate_9points_dataset(modelclass, model_path, test_input_name_prefix):
  model = modelclass.build_model()
  model.load_weights(model_path)
  logger.info("Model loaded")

  face_points, left_eyes, right_eyes, ys = load_9point_dataset(test_input_name_prefix)
  print(model.evaluate([left_eyes, right_eyes, face_points], ys))
