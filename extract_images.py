import cv2
import logging
import json
import os
from argparse import ArgumentParser

# Setup logger.
logging.basicConfig(level='INFO')
logger = logging.getLogger('extract_images')

# Parse command line arguments.
parser = ArgumentParser()
parser.add_argument('input_name',
        help='Name of input_video/event_log file (without extension)')
parser.add_argument('output_folder',
        help='Name of folder for output files')
args = parser.parse_args()

# Open the input video file
input_video = cv2.VideoCapture(args.input_name + ".avi")
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
logger.info("Input video width={}, height={}".format(width, height))

# Read event log
with open(args.input_name + ".jsonl") as event_log:
    events = list(map(json.loads, event_log.readlines()))
num_events = len(events)

frame_count = 0
current_event = 0
while True:
    ret, frame = input_video.read()
    # Discard last few events. User was maybe not concentrated and decided to
    # stop recording, so it makes sense to discard them.
    if not ret or current_event + 5 >= num_events:
        break
    start_time = events[current_event]["frame_count"]
    end_time = events[current_event + 1]["frame_count"]
    frame_to_extract = start_time + (end_time - start_time) * 0.75
    if frame_count > frame_to_extract:
        name = "{}_{}".format(os.path.join(args.output_folder,
            os.path.basename(args.input_name)), current_event)
        cv2.imwrite(name + ".jpg", frame)
        with open(name + ".json", "w") as event_file:
            event_file.write(json.dumps(events[current_event]) + "\n")
        current_event += 1
    frame_count += 1

logger.info('Frame count: {}'.format(frame_count))

# All done!
input_video.release()
cv2.destroyAllWindows()
