import mediapipe
import cv2
import numpy
import uuid
import os

class base:
    def __init__(
        self, video_feed:int,
        win_title:str,
        min_detection_confidence:float = 0.6, min_tracking_confidence:float = 0.3,
        show_log:bool = False
        ):
        self._VIDEO_FEED = video_feed
        self._WIN_TITLE = win_title
        self._MIN_DETECTION_CONFIDENCE = min_detection_confidence
        self._MIN_TRAKING_CONFIDENCE = min_tracking_confidence
        self.show_log = show_log

        self.__mp_drawing = mediapipe.solutions.drawing_utils
        self.__mp_hands = mediapipe.solutions.hands
        self.__cap = cv2.VideoCapture(self._VIDEO_FEED)

    def start(self) -> None:
        with self.__mp_hands.Hands(
            min_detection_confidence= self._MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence= self._MIN_TRAKING_CONFIDENCE
        ) as hands:

            while self.__cap.isOpened():
                ret, frame = self.__cap.read()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = cv2.flip(image, 1)
                image.flags.writeable = False

                results = hands.process(image)
                image.flags.writeable = True

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if self.show_log:
                    print(results.multi_hand_landmarks)

                if results.multi_hand_landmarks:
                            for num, hand in enumerate(results.multi_hand_landmarks):
                                self.__mp_drawing.draw_landmarks(
                                    image, hand, self.__mp_hands.HAND_CONNECTIONS, 
                                    self.__mp_drawing.DrawingSpec(color=(0, 0, 225), thickness=2, circle_radius=1),
                                    self.__mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                    )

                cv2.imshow(self._WIN_TITLE, image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    
        self.__cap.release()
        cv2.destroyWindow(self._WIN_TITLE)