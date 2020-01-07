import cv2
import datetime
import logging
import numpy as np
import random
import screeninfo
import json
from argparse import ArgumentParser

# Setup logger.
logging.basicConfig(level='INFO')
logger = logging.getLogger('record')

def random_strategy(width, height):
    x = random.randrange(5, width - 5)
    y = random.randrange(5, height - 5)
    return (x, y), (x, y)

class LeftRightCenteredStrategy:
    def __init__(self):
        self.last_left = True

    def __call__(self, width, height):
        self.last_left = not self.last_left
        if self.last_left:
            return "right", (width - 50, height / 2)
        return "left", (50, height / 2)

class NinePointsStrategy:
    def __init__(self):
        self.last_point_label = 0

    def __call__(self, width, height):
        xs = [50, width / 2, width - 50]
        ys = [50, height / 2, height - 50]
        while True:
            x_index = random.randrange(0, 3)
            y_index = random.randrange(0, 3)
            label = y_index * 3 + x_index
            if label == self.last_point_label:
                continue
            self.last_point_label = label
            return label, (xs[x_index], ys[y_index])

class DotGenerator:
    def __init__(self, strategy, image, events_log, start_time):
        self.image = image
        self.height = len(image)
        self.width = len(image[0])
        self.events_log = events_log
        self.next_event = start_time + datetime.timedelta(seconds=1)
        self.strategy = strategy
        # Values does not really matter since there is no dot yet.
        self.last_dot = (self.width / 2, self.height / 2)

    def get_image(self, current_time, frame_count):
        if current_time >= self.next_event:
            self.next_event = current_time + datetime.timedelta(seconds=1)
            cv2.circle(self.image, self.last_dot, 5, (255, 255, 255), -1)
            label, self.last_dot = self.strategy(self.width, self.height)
            cv2.circle(self.image, self.last_dot, 5, (0, 0, 0), -1)
            event = {
                "frame_count": frame_count,
                "timepoint": str(current_time),
                "label": label,
                "x": self.last_dot[0], "y": self.last_dot[1],
                "width": self.width, "height": self.height
            }
            self.events_log.write(json.dumps(event) + "\n")
        return self.image

DOT_GENERATORS = {"lr_centered": LeftRightCenteredStrategy(),
                  "random": random_strategy,
                  "9points": NinePointsStrategy()}

# Parse command line arguments.
parser = ArgumentParser()
parser.add_argument('output_name',
        help='Name of output files (without extension)')
parser.add_argument('dot_generator',
        choices=list(DOT_GENERATORS),
        help='Dot generator class to be used')
parser.add_argument('--camera_id', type=int, default=0,
        help='Camera id')
parser.add_argument('--screen_id', type=int, default=0,
        help='Screen id of monitor with camera')
args = parser.parse_args()

# Open the camera "file".
video_capture = cv2.VideoCapture(args.camera_id)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
logger.info('Camera capture width={} height={}'.format(width, height))

# Create output files.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# Camera FPS is not fixed, but variable depending on brightness of the scene.
# Since we only care about storing the video and not how it looks we can set
# FPS to anything.
output_video = cv2.VideoWriter(
        args.output_name + '.avi', fourcc, 12, (width, height))
events_log = open(args.output_name + '.jsonl', 'w')

# Create image to be shown
screen = screeninfo.get_monitors()[args.screen_id]
logger.info('Screen width={} height={}'.format(screen.width, screen.height))
image = np.zeros([screen.height, screen.width, 3])
image.fill(255)
cv2.putText(image,
        "Wait for dots to appear and look at them. "
        "When you become bored press 'q' to quit",
        (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

dot_generator = DotGenerator(DOT_GENERATORS[args.dot_generator],
        image, events_log, datetime.datetime.now())

WINDOW = "window"
cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

frame_count = 0
while True:
    _, frame = video_capture.read()
    output_video.write(frame)
    frame_count += 1
    cv2.imshow(WINDOW, dot_generator.get_image(
            datetime.datetime.now(), frame_count))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

logger.info('Frame count: {}'.format(frame_count))

# All done!
video_capture.release()
cv2.destroyAllWindows()
events_log.close()
