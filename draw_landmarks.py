from PIL import Image, ImageDraw
import face_recognition
from argparse import ArgumentParser

# Parse command line arguments.
parser = ArgumentParser()
parser.add_argument('input_image',
        help='Name of input_image file')
args = parser.parse_args()


# Load the jpg file into a numpy array
image = face_recognition.load_image_file(args.input_image)

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = Image.fromarray(image)
for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    for landmark_name in face_landmarks:
        for dot in face_landmarks[landmark_name]:
            d.ellipse((dot[0] - 2, dot[1] - 2, dot[0] + 2, dot[1] + 2),
                    fill=(255, 255, 255, 255))

pil_image.show()
