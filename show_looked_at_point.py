import cv2
import datetime
import logging
import numpy as np
import random
import screeninfo
import json
import ml
from argparse import ArgumentParser
from extract_face_eye_data import extract_face_eye_data

# Setup logger.
logging.basicConfig(level='INFO')
logger = logging.getLogger('record')

class ImageDrawer:
    def __init__(self, screen_id):
        screen = screeninfo.get_monitors()[screen_id]
        self.height = screen.height
        self.width = screen.width
        self.image = np.zeros([self.height, self.width, 3], dtype="uint8")
        self.image.fill(255)
        # Values does not really matter since no dot is looked at the start
        self.last_dot = (self.width / 2, self.height / 2)

    def select_dot(self, dot_index, frame):
        cv2.circle(self.image, self.last_dot, 20, (255, 255, 255), -1)

        new_width = int(self.image.shape[1] * 0.25)
        new_height = int(float(new_width) / frame.shape[1] * frame.shape[0])
        mini_frame = cv2.resize(frame, (new_width, new_height))
        self.image[self.height - new_height:, self.width - new_width:] = mini_frame[:, :]

        xs = [50, self.width / 2, self.width - 50]
        ys = [50, self.height / 2, self.height - 50]
        for x in xs:
            for y in ys:
                cv2.circle(self.image, (x, y), 15, (0, 0, 0), -1)

        cv2.circle(self.image, self.last_dot, 15, (0, 0, 0), -1)
        x = xs[dot_index % 3]
        y = ys[dot_index / 3]
        self.last_dot = (x, y)

        cv2.circle(self.image, self.last_dot, 20, (0, 0, 255), -1)
        return self.image

# Parse command line arguments.
parser = ArgumentParser()
parser.add_argument('model_checkpoints',
        help='Name of output files (without extension)')
parser.add_argument('--camera_id', type=int, default=0,
        help='Camera id')
parser.add_argument('--screen_id', type=int, default=0,
        help='Screen id of monitor with camera')
args = parser.parse_args()

# Open the camera "file".
video_capture = cv2.VideoCapture(args.camera_id)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
logger.info('Camera capture width={} height={}'.format(width, height))

WINDOW = "window"
cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

model = ml.Chaos2Pool.build_model()
model.load_weights(args.model_checkpoints)
logger.info("Model loaded")

image_drawer = ImageDrawer(args.screen_id)

while True:
    _, frame = video_capture.read()
    # Convert from BGR (opencv) to RGB (face_recognition)
    dlib_frame = frame[:, :, ::-1]
    data = extract_face_eye_data(dlib_frame, do_it_faster=True)
    if data is None: continue
    face_landmarks, eye_images = data

    # Convert from pil to tensorflow format
    eye_images = eye_images[1:]
    left_eye_pil_image = eye_images[0][1]
    right_eye_pil_image = eye_images[1][1]
    left_eye = np.array(left_eye_pil_image)[:, :, 0:3] / 255.
    right_eye = np.array(right_eye_pil_image)[:, :, 0:3] / 255.

    points = []
    for landmark in sorted(face_landmarks):
        for x, y in face_landmarks[landmark]:
            points.append((x / float(width), y / float(height)))
    points = np.array(points)

    probabilities = model.predict([np.array([left_eye]),
        np.array([right_eye]), np.array([points])])
    dot_index = np.argmax(probabilities)

    cv2.imshow(WINDOW, image_drawer.select_dot(dot_index, frame))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# All done!
video_capture.release()
cv2.destroyAllWindows()
