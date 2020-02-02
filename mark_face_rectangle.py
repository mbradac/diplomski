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

def mark_rectangle(image):
    face_landmark_list = face_recognition.face_landmarks(image)
    assert face_landmark_list

    min_x, min_y, max_x, max_y = 100000, 100000, 0, 0
    for face_landmarks in face_landmark_list:
        for landmark in face_landmarks:
            for x, y in face_landmarks[landmark]:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    dimension = max(max_x - min_x, max_y - min_y)
    x_center = (min_x + max_x) // 2
    y_center = (min_y + max_y) // 2
    min_x = x_center - dimension // 2
    max_x = x_center + (dimension + 1) // 2
    min_y = y_center - dimension // 2
    max_y = y_center + (dimension + 1) // 2

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=3)
    return pil_image


if __name__ == "__main__":
    # Setup logger.
    logging.basicConfig(level='INFO')
    logger = logging.getLogger('extract_face_eye_data')

    # Parse command line arguments.
    parser = ArgumentParser()
    parser.add_argument('input_name', help='Names of input_image')
    parser.add_argument('output_name', help='Name of output image')
    args = parser.parse_args()

    image = face_recognition.load_image_file(args.input_name)
    pil_image = mark_rectangle(image)
    pil_image.save(args.output_name)
