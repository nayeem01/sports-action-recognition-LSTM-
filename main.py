import cv2 as cv
import mediapipe as mp
from keras.models import model_from_json
import numpy as np
import tensorflow as tf
from numba import cuda

device = cuda.get_current_device()
device.reset()

# configuring GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)  # Draw face connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


class VideoWindow:
    def __init__(self, videoPath, scale):
        self.videoPath = videoPath
        self.scale = scale
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.threshold = 0.5
        self.actions = np.array(['SkateBoarding', 'Run-Side'])

    def rescaleFrame(self, frame):
        width = int(frame.shape[1] * self.scale)
        height = int(frame.shape[0] * self.scale)

        dimensions = (width, height)

        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    def mediapipe_detection(self, image, model):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        # Draw face connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(
                                      color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(
                                      color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
        # Draw right hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
        ) if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
        ) if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
        ) if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
        ) if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def loadingVideo(self):
        cap = cv.VideoCapture(self.videoPath)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                _, frame = cap.read()

                frame = self.rescaleFrame(frame)

                # cv.putText(frame, "Action : ",
                #            (40, 40),  font, 1, (0, 255, 0), 2, cv.LINE_AA)

                image, results = self.mediapipe_detection(frame, pose)

                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-70:]

                if len(self.sequence) == 70:
                    res = actions_model.predict(
                        np.expand_dims(self.sequence, axis=0))[0]
                    # print(self.actions[np.argmax(res)])
                    print(res)
                    self.predictions.append(np.argmax(res))

                    str_classname = "SkateBoarding : {:.3f}".format(
                        res[0])
                    cv.putText(image, str_classname,
                               (40, 40),  font, 1, (0, 255, 0), 2, cv.LINE_AA)

                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())

                cv.imshow('sports actions', image)

                if cv.waitKey(1) & 0xFF == ord('d'):
                    break

            cap.release()
            cv.destroyAllWindows()


if __name__ == "__main__":

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    # Loading the model
    json_file = open('actions.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    actions_model = model_from_json(loaded_model_json)
    actions_model.load_weights("actions.h5")
    font = cv.FONT_HERSHEY_SIMPLEX
    video = VideoWindow('test/test2.mp4', 0.3)
    video.loadingVideo()
