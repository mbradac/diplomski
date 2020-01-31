import cv2
import logging
import json
import face_recognition
import glob
import os
from argparse import ArgumentParser
from PIL import Image, ImageDraw
import sys
import numpy as np

EYE_IMAGE_SIZE = 64


def extract_face_eye_data(image, context_pixels=0, do_it_faster=False):
    if do_it_faster:
        RESIZE_FACTOR = 0.6
        smaller_image = Image.fromarray(image)
        new_width = int(image.shape[1] * RESIZE_FACTOR)
        new_height = int(image.shape[0] * RESIZE_FACTOR)
        smaller_image = smaller_image.resize((new_width, new_height))
        smaller_image = np.array(smaller_image)
        face_landmark_list = face_recognition.face_landmarks(
                smaller_image)
        for face_landmarks in face_landmark_list:
            for landmark in face_landmarks:
                for i, point in enumerate(face_landmarks[landmark]):
                    x, y = point
                    face_landmarks[landmark][i] = (
                            int(x / RESIZE_FACTOR),
                            int(y / RESIZE_FACTOR))
    else:
        face_landmark_list = face_recognition.face_landmarks(image)
    if not face_landmark_list: return None

    def extract(min_x, min_y, max_x, max_y, context_pixels):
        dimension = max(max_x - min_x, max_y - min_y) + context_pixels
        x_center = (min_x + max_x) // 2
        y_center = (min_y + max_y) // 2
        min_x = x_center - dimension // 2
        max_x = x_center + (dimension + 1) // 2
        min_y = y_center - dimension // 2
        max_y = y_center + (dimension + 1) // 2
        pil_image = Image.fromarray(image[min_y:max_y, min_x:max_x])
        pil_image = pil_image.resize(
                (EYE_IMAGE_SIZE, EYE_IMAGE_SIZE), Image.BILINEAR)
        return pil_image

    face_min_x, face_min_y, face_max_x, face_max_y = 100000, 100000, 0, 0
    for face_landmarks in face_landmark_list:
        for landmark in face_landmarks:
            for x, y in face_landmarks[landmark]:
                face_min_x = min(face_min_x, x)
                face_min_y = min(face_min_y, y)
                face_max_x = max(face_max_x, x)
                face_max_y = max(face_max_y, y)

    images = [("face", extract(face_min_x, face_min_y, face_max_x,
        face_max_y, context_pixels))]

    for eye_name in ('left_eye', 'right_eye'):
        eye_dots = face_landmark_list[0][eye_name]
        xs = list(map(lambda x: x[0], eye_dots))
        ys = list(map(lambda y: y[1], eye_dots))
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        images.append((eye_name,
            extract(min_x, min_y, max_x, max_y, context_pixels)))

    return face_landmark_list[0], images


if __name__ == "__main__":
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

    for i, input_name in enumerate(input_names):
        if i % 100 == 0:
            logger.info("Processing image {}/{}".format(i, len(input_names)))

        image = face_recognition.load_image_file(input_name + '.jpg')
        for context_pixels in (10, 5, 0, -5):
            data = extract_face_eye_data(image, context_pixels)
            if data is None: continue
            face_landmarks, images = data

            with open(input_name + '.json') as fp:
                image_data = json.load(fp)

            name = os.path.join(args.output_folder,
                    "context{}__".format(context_pixels) +
                        os.path.basename(input_name))
            image_data['landmarks'] = face_landmarks
            for feature_name, pil_image in images:
                with open(name + '.json', 'w') as fp:
                    fp.write(json.dumps(image_data) + '\n')
                pil_image.save(name + '_' + feature_name[0] + '.jpg')
