import logging
import glob
import os
import json
import numpy as np
from model import build_model
from argparse import ArgumentParser
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Setup logger.
logging.basicConfig(level='INFO')
logger = logging.getLogger('train')

model = build_model()

# Parse command line arguments.
parser = ArgumentParser()
parser.add_argument('input_name_prefix',
        help='Names of input_images/json files (without extensions), all '
        'files matching regex input_name_prefix* will be considered input')
parser.add_argument('trained_model_path',
        help='Path where the trained model will be saved')
args = parser.parse_args()

input_names = glob.glob(args.input_name_prefix + '*')
input_names = filter(lambda x: os.path.splitext(x)[1] != '.jpg', input_names)
input_names = list(map(lambda x: os.path.splitext(x)[0], input_names))

face_points = []
left_eyes = []
right_eyes = []
ys = []

for input_name in input_names:
    with open(input_name + '.json') as fp:
        data = json.load(fp)
        height = data["height"]
        width = data["width"]
        ys.append([data["x"] / width, data["y"] / height])
        points = []
        for landmark in sorted(data["landmarks"]):
            for x, y in data["landmarks"][landmark]:
                points.append((x / width, y / height))
        face_points.append(points)
    with open(input_name + '_l.jpg') as fp:
        left_eyes.append(img_to_array(load_img(input_name + '_l.jpg')))
        right_eyes.append(img_to_array(load_img(input_name + '_r.jpg')))

#print(face_points)
#print(left_eyes)
#print(right_eyes)
#print(ys)

size = len(face_points)
train_size = int(0.8 * size)

face_points = np.array(face_points)
left_eyes = np.array(left_eyes)
right_eyes = np.array(right_eyes)
ys = np.array(ys)

face_points_train, face_points_test = \
        face_points[:train_size], face_points[train_size:]
left_eyes_train, left_eyes_test = \
        left_eyes[:train_size], left_eyes[train_size:]
right_eyes_train, right_eyes_test = \
        right_eyes[:train_size], right_eyes[train_size:]
ys_train, ys_test = ys[:train_size], ys[train_size:]

model.fit([left_eyes_train, right_eyes_train, face_points_train],
        [ys_train],
#        batch_size=128,
        batch_size=1,
#        epochs=100,
        epochs=30,
        verbose=1,
        validation_data=(
            [left_eyes_test, right_eyes_test, face_points_test], [ys_test]))

model.save(args.trained_model_path)

print(model.predict([left_eyes_test[:1], right_eyes_test[:1], face_points_test[:1]]))
print(ys_test[:1])
print(model.predict([left_eyes_train[:1], right_eyes_train[:1], face_points_train[:1]]))
print(ys_train[:1])
