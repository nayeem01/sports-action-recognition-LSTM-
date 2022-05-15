import cv2 as cv
import mediapipe as mp


# def holisticPos():
class VideoWindow:
    def __init__(self, videoPath, scale):
        self.videoPath = videoPath
        self.scale = scale

    def rescaleFrame(self, frame):
        width = int(frame.shape[1] * self.scale)
        height = int(frame.shape[0] * self.scale)

        dimensions = (width, height)

        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    def loadingVideo(self):
        cap = cv.VideoCapture(self.videoPath)
        with mp_pose.Pose(static_image_mode=True,
                          model_complexity=2,
                          enable_segmentation=True,
                          min_detection_confidence=0.5) as pose:
            while cap.isOpened():
                _, frame = cap.read()

                frame = self.rescaleFrame(frame)
                landmarks = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(
                    frame,
                    landmarks.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
                cv.imshow('skating', frame)

                if cv.waitKey(1) & 0xFF == ord('d'):
                    break

            cap.release()
            cv.destroyAllWindows()


if __name__ == "__main__":

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    video = VideoWindow('test/test1.mp4', 0.3)
    video.loadingVideo()
