"""
    use "q" for capturing new gesture
    you can change caps variables
"""
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import time

FLAG = 0
VIDEO_FEED = 0
WIN_TITLE = "HELLO_WORLD"
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRAKING_CONFIDENCE = 0.3

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(VIDEO_FEED)

try:
    os.mkdir('data')
except:
    pass

with mp_hands.Hands(
    min_detection_confidence= MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence= MIN_TRAKING_CONFIDENCE
    ) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        
        results = hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        print(results.multi_hand_landmarks)

        cv2.imshow(WIN_TITLE, image)

        try:
            os.mkdir("data/"+str(flag))
            WIN_TITLE = "capturing gesture" + str(flag)
        except:
            pass

        if results.multi_hand_landmarks:
            time.sleep(0.1)
            cv2.imwrite(os.path.join('data',f"{flag}",f"{uuid.uuid1()}.jpg"), image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            FLAG += 1
            print(f"capturing gestures in 3 sec.")
            time.sleep(3)

cap.release()
cv2.destroyAllWindows()