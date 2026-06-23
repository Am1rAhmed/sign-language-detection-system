import cv2
import numpy as np
import os
import mediapipe as mp
from utils import (mediapipe_detection,draw_styled_landmarks,extract_keypoints)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Path for exported data, numpy arrays
DATA_PATH = os.path.join("MP_Data")

# Actions that we try to detect
actions = np.array(['Hello', 'Thanks', 'I love you','Help','Sorry','Eat','Nice to meet you'])




# The number of videos we want to collect per sign
no_sequences = 85
# Videos are going to be 30 frames in length
sequence_length = 45

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
#Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            #Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                #Read feed
                ret, frame = cap.read()

                #Make detections
                image, results = mediapipe_detection(frame, holistic)
                # print(results)

                #Draw landmarks
                draw_styled_landmarks(image, results)

                #Apply collection logic
                if frame_num == 0:
                    cv2.putText(image, "STARTING COLLECTION", (120,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', 
                                (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 0), 1, cv2.LINE_AA)
                    cv2.waitKey(3000)
                else:                 
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', 
                                (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 0), 1, cv2.LINE_AA)
                # New export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                #Show feed
                cv2.imshow("OpenCV Feed", image)

                #Break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()