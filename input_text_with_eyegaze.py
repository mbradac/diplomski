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
from collections import deque
from collections import Counter
from text_input_automata import TextInputAutomata

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

    def get_image(self, dot_index, do_input_point,
            central_text, dot_labels, frame):
        self.image.fill(255)
        #cv2.circle(self.image, self.last_dot, 15, (255, 255, 255), -1)

        new_width = int(self.image.shape[1] * 0.25)
        new_height = int(float(new_width) / frame.shape[1] * frame.shape[0])
        mini_frame = cv2.resize(frame, (new_width, new_height))
        self.image[self.height - new_height:, self.width - new_width:] = \
                mini_frame[:, :]

        xs = [50, self.width / 2, self.width - 50]
        ys = [50, self.height / 2, self.height - 50]
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                i = yi * 3 + xi
                xshift = [0, 0, -50][xi]
                yshift = [40, 40, -30][yi]
                cv2.circle(self.image, (x, y), 15, (0, 0, 0), -1)
                cv2.putText(self.image, dot_labels[i],
                        (x + xshift, y + yshift),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(self.image, "Uneseni tekst: " + central_text,
                (self.width / 2 - 400, self.height / 2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        #cv2.circle(self.image, self.last_dot, 5, (0, 0, 0), -1)
        x = xs[dot_index % 3]
        y = ys[dot_index / 3]
        #self.last_dot = (x, y)

        dot_color = (0, 255, 0) if do_input_point else (0, 0, 255)
        cv2.circle(self.image, (x, y), 20, dot_color, -1)
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
selected_points = deque()
text_input_automata = TextInputAutomata()
text = ""

while True:
    _, frame = video_capture.read()
    current_time = datetime.datetime.now()
    # Convert from BGR (opencv) to RGB (face_recognition)
    dlib_frame = frame[:, :, ::-1]
    data = extract_face_eye_data(dlib_frame, do_it_faster=True)
    if data is None:
        selected_points.append((current_time, -1))
        continue
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

    selected_points.append((current_time, dot_index))
    # Don't keep more than 2 seconds worth of looked at dots history
    TIME_WINDOW_SIZE = datetime.timedelta(seconds=2)
    while len(selected_points) and \
            selected_points[-1][0] - selected_points[0][0] > TIME_WINDOW_SIZE:
        selected_points.popleft()

    # If it passed more than MINIMUM_INPUT_TIME time from last input and
    # in history of looked at points most common point appears with
    # MINIMUM_INPUT_FREQUENCY frequency than treat it as a input.
    MINIMUM_INPUT_FREQUENCY = 0.7
    MINIMUM_INPUT_TIME = datetime.timedelta(seconds=1)
    most_common_dot, num_most_common = Counter(
            map(lambda x: x[1], selected_points)).most_common(1)[0]
    if (num_most_common >
            len(selected_points) * MINIMUM_INPUT_FREQUENCY and
            selected_points[-1][0] - selected_points[0][0] >
            MINIMUM_INPUT_TIME and
            most_common_dot == dot_index):
        text = text_input_automata.transition(dot_index, text)
        do_input_point = True
        selected_points.clear()
    else:
        do_input_point = False

    cv2.imshow(WINDOW, image_drawer.get_image(
        dot_index, do_input_point, text, text_input_automata.labels(), frame))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# All done!
video_capture.release()
cv2.destroyAllWindows()
