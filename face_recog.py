import face_recognition
import os
from cv2 import cv2

TRAIN_DIR = "train_images"  # directory for training faces
TEST_DIR = "test_images"  # directory for testing faces

TOLERANCE = 0.6  # lower for higher accuracy, higher for more matches but poorer accuracy
FRAME_THICKNESS = 3  # thickness of the frame around detected image in pixels
FONT_THICKNESS = 2
MODEL = 'hog'  # cnn or hog


known_faces = []
known_names = []


def load_known_faces(path):
    faces, names = [], []
    for name in os.listdir(path):
        # Next we load every file of faces of known person
        for filename in os.listdir(f'{path}/{name}'):
            # Load an image
            image = face_recognition.load_image_file(
                f'{path}/{name}/{filename}')
            try:
                encoding = face_recognition.face_encodings(image)[0]
            except:
                print(f"Could not find a face in {filename}")
            # Append encodings and name
            faces.append(encoding)
            names.append(name)
    return faces, names


def process_unknown_faces(path):
    print('Processing unknown faces...')
    for filename in os.listdir(path):
        # Load image
        print(f'Filename {filename}', end='')
        image = face_recognition.load_image_file(
            f'{path}/{filename}')

        locations = face_recognition.face_locations(image, model=MODEL)

        encodings = face_recognition.face_encodings(image, locations)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        print(f', found {len(encodings)} face(s)')
        for face_encoding, face_location in zip(encodings, locations):

            results = face_recognition.compare_faces(
                known_faces, face_encoding, TOLERANCE)

            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                match = known_names[results.index(True)]
                print(f"Match found! {match}")

                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                color = [0, 255, 0]

                cv2.rectangle(image, top_left, bottom_right,
                              color, FRAME_THICKNESS)

                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)

                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

        # Show image
        cv2.imshow(filename, image)
        cv2.waitKey(0)
        cv2.destroyWindow(filename)


known_faces, known_names = load_known_faces(TRAIN_DIR)
process_unknown_faces(TEST_DIR)
