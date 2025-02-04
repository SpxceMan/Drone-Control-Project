import cv2 as cv
import mediapipe as mp
import HandTrackingModule as htmod
import time
import numpy as np
import vgamepad as vg
#############################################################
wCam, hCam = 640, 480 #width and height of the camera
#############################################################

detector = htmod.handDetector(detectionCon=0.7)

gamepad = vg.VX360Gamepad()

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

prevTime = 0
currTime = 0
fps = 0

while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)

    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) == 0:
    # if no fingers detected, set both joysticks to 0
        x_joy = y_joy = x_joy_right = y_joy_right = 0#if there are landmarks
        # print(lmList[4], lmList[8])
    else:   
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        x3, y3 = lmList[12][1], lmList[12][2]
        x4, y4 = lmList[16][1], lmList[16][2]
        x5, y5 = lmList[20][1], lmList[20][2]

        # distance = np.hypot(x2-x1, y2-y1)
        # print(distance)q\
        # cx, cy = (x1+x2)//2, (y1+y2)//2

        cv.circle(frame, (x1, y1), 15, (255, 0, 255), -1)
        cv.circle(frame, (x2, y2), 15, (255, 0, 255), -1)
        cv.circle(frame, (x3, y3), 15, (255, 0, 255), -1)
        cv.circle(frame, (x4, y4), 15, (255, 0, 255), -1)
        cv.circle(frame, (x5, y5), 15, (255, 0, 255), -1)

        cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv.line(frame, (x1, y1), (x3, y3), (255, 0, 0), 3)
        cv.line(frame, (x1, y1), (x4, y4), (255, 0, 0), 3)
        cv.line(frame, (x1, y1), (x5, y5), (255, 0, 0), 3)

        max_dist = 530
        min_dist = 50

        distance_x = np.hypot(x2-x1, y2-y1)
        distance_x = np.clip(distance_x, min_dist, max_dist)

        x_joy = int(np.interp(distance_x, [min_dist, max_dist], [-32767, 32767]))

        

        distance_y = np.hypot(x3-x1, y3-y1)
        distance_y = np.clip(distance_y, min_dist, max_dist)

        y_joy = int(np.interp(distance_y, [min_dist, max_dist], [32767, -32767]))

        print(int(distance_x), x_joy, int(distance_y), y_joy)

        distance_x_right = np.hypot(x4 - x1, y4 - y1)
        distance_y_right = np.hypot(x5 - x1, y5 - y1)

        distance_x_right = np.clip(distance_x_right, min_dist, max_dist)
        distance_y_right = np.clip(distance_y_right, min_dist, max_dist)

        x_joy_right = int(np.interp(distance_x_right, [min_dist, max_dist], [-32767, 32767]))
        y_joy_right = int(np.interp(distance_y_right, [min_dist, max_dist], [32767, -32767]))


        print(distance_x_right, x_joy_right, distance_y_right, y_joy_right)


        gamepad.right_joystick(x_joy_right, y_joy_right)
        gamepad.update()
        gamepad.left_joystick(x_joy, y_joy)
        gamepad.update()



    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # frame = cv.resize(frame, (640, 480))
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()