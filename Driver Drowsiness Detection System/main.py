from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from pygame import mixer

#config
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 35
YAWN_THRESH = 30
YAWN_CONSEC_FRAMES = 30

#globals
alarm_status = False
alarm_status2 = False
COUNTER = 0
YAWN_COUNTER = 0

#init audio
mixer.init()
#download audio
sound1 = mixer.Sound(r"C:\Driver Drowsiness Detection System\wake_up.mp3")  
# drowsiness alarm
sound2 = mixer.Sound(r"C:\Driver Drowsiness Detection System\warn.mp3")     
# yawn alarm

def alarm():
    """play alarm sounds depending on which status is active"""
    global alarm_status, alarm_status2
    while alarm_status:
        print("[eye alarm] wake up!")
        sound1.play()
        time.sleep(sound1.get_length())  # wait until sound done
    if alarm_status2:
        print("[yawn alarm] take some fresh air")
        sound2.play()
        time.sleep(sound2.get_length())

def eye_aspect_ratio(eye):
    """calculate eye aspect ratio from 6 landmarks"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_ear(shape):
    """return avg EAR and eye coords"""
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return ( (leftEAR + rightEAR) / 2.0, leftEye, rightEye )

def lip_distance(shape):
    """measure vertical mouth gap"""
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1])

# args
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

#detectors
print("loading detectors...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#start video
print("starting video...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)  # warmup

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, leftEye, rightEye = final_ear(shape)
        mouth_gap = lip_distance(shape)

        # draw eyes + lips
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [shape[48:60]], -1, (0, 255, 0), 1)

        # drowsiness detection
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES and not alarm_status:
                alarm_status = True
                Thread(target=alarm, daemon=True).start()
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        # yawn detection
        if mouth_gap > YAWN_THRESH:
            YAWN_COUNTER += 1
            if YAWN_COUNTER >= YAWN_CONSEC_FRAMES and not alarm_status2:
                alarm_status2 = True
                Thread(target=alarm, daemon=True).start()
            cv2.putText(frame, "YAWN ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            YAWN_COUNTER = 0
            alarm_status2 = False

        #HUD: show current EAR and mouth gap
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"YAWN: {mouth_gap:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
