import cv2
import logging
import json
import face_recognition
import glob
import os
from argparse import ArgumentParser
from PIL import Image, ImageDraw
import sys

EYE_IMAGE_SIZE = 64

# Setup logger.
logging.basicConfig(level='INFO')
logger = logging.getLogger('extract_face_eye_data')

# Parse command line arguments.
parser = ArgumentParser()
parser.add_argument('input_name_prefix',
        help='Names of input_image/event_log files (without extensions), all '
        'files matching regex input_name_prefix* will be considered input')
parser.add_argument('output_folder',
        help='Name of folder for output files')
args = parser.parse_args()

input_names = set(map(lambda x: os.path.splitext(x)[0],
        glob.glob(args.input_name_prefix + '*')))

for input_name in input_names:
    image = face_recognition.load_image_file(input_name + '.jpg')
    with open(input_name + '.json') as fp:
        image_data = json.load(fp)
    face_landmark_list = face_recognition.face_landmarks(image)
    if not face_landmark_list: continue

    name = os.path.join(args.output_folder, os.path.basename(input_name))
    image_data['landmarks'] = face_landmark_list[0]
    with open(name + '.json', 'w') as fp:
        fp.write(json.dumps(image_data) + '\n')

    for eye_name in ('left_eye', 'right_eye'):
        eye_dots = face_landmark_list[0][eye_name]
        xs = list(map(lambda x: x[0], eye_dots))
        ys = list(map(lambda y: y[1], eye_dots))
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        x_shift = (max_x - min_x + 1) // 2
        min_x -= x_shift
        max_x += x_shift
        dimension = max_x - min_x
        y_center = (min_y + max_y) // 2
        min_y = y_center - dimension // 2
        max_y = y_center + (dimension + 1) // 2
        pil_image = Image.fromarray(image[min_y:max_y, min_x:max_x])
        pil_image = pil_image.resize(
                (EYE_IMAGE_SIZE, EYE_IMAGE_SIZE), Image.BILINEAR)
        pil_image.save(name + '_' + eye_name[0] + '.jpg')
