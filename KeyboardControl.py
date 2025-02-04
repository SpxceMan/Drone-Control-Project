import cv2 as cv
import time
import numpy as np
import handTrackingModule as htm
import math
from pynput.keyboard import Controller, Key


wCam, hCam = 620, 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
keyboard = Controller()
detector = htm.handDetector(detectionCon=0.8)

pTime = 0
w_pressed = False 
d_pressed = False 
u_a_pressed = False
d_a_pressed = False

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        x3, y3 = lmlist[20][1], lmlist[20][2]
        x4, y4 = lmlist[16][1], lmlist[16][2]
        x5, y5 = lmlist[12][1], lmlist[12][2]
        cx1, cy1 = (x1 + x2) // 2, (y1 + y2) // 2
        cx2, cy2 = (x1 + x3) // 2, (y1 + y3) // 2
        cx3, cy3 = (x1 + x4) // 2, (y1 + y4) // 2
        cx4, cy4 = (x1 + x5) // 2, (y1 + y5) // 2

        # cv.circle(img, (x1, y1), 10, (255, 0, 0), -1)
        # cv.circle(img, (x2, y2), 10, (255, 0, 0), -1)
        # cv.circle(img, (x3, y3), 10, (255, 0, 0), -1)
        # cv.circle(img, (x4, y4), 10, (255, 0, 0), -1)
        # cv.circle(img, (x5, y5), 10, (255, 0, 0), -1)
        # cv.circle(img, (cx1, cy1), 10, (255, 0, 0), -1)
        # cv.circle(img, (cx2, cy2), 10, (255, 0, 0), -1)
        # cv.circle(img, (cx3, cy3), 10, (255, 0, 0), -1)
        # cv.circle(img, (cx4, cy4), 10, (255, 0, 0), -1)
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv.line(img, (x1, y1), (x3, y3), (0, 255, 0), 2)
        cv.line(img, (x1, y1), (x4, y4), (0, 0, 255), 2)
        cv.line(img, (x1, y1), (x5, y5), (255, 0, 255), 2)

        length_ti = math.hypot((x2 - x1), (y2 - y1))
        length_tl = math.hypot((x3 - x1), (y3 - y1))
        length_tr = math.hypot((x4 - x1), (y4 - y1))
        length_tm = math.hypot((x5 - x1), (y5 - y1))
        #print(f"Distance: {length:.1f}")

        #w presss
        if length_ti < 20:
            cv.circle(img, (cx1, cy1), 10, (0, 255, 0), -1)
            if not w_pressed:
                keyboard.press("w")
                print("W PRESSED")
                w_pressed = True
        elif w_pressed:
            keyboard.release("w")
            print("W RELEASED")
            w_pressed = False

        #D press
        if length_tl < 20:
            cv.circle(img, (cx2, cy2), 10, (0, 255, 0), -1)
            if not d_pressed:
                keyboard.press("d")
                print("D PRESSED")
                d_pressed = True
        elif d_pressed:
            keyboard.release("d")
            print("D RELEASED")
            d_pressed = False

        #Up arrow fr
        if length_tr < 20:
            cv.circle(img, (cx3, cy3), 10, (0, 255, 0), -1)
            if not u_a_pressed:
                keyboard.press(Key.up)
                print("UP PRESSED")
                u_a_pressed = True
        elif u_a_pressed:
            keyboard.release(Key.up)
            print("UP RELEASED")
            u_a_pressed = False

        #dwn arrow fr
        if length_tm < 20:
            cv.circle(img, (cx4, cy4), 10, (0, 255, 0), -1)
            if not d_a_pressed:
                keyboard.press(Key.down)
                print("DOWN PRESSED")
                d_a_pressed = True
        elif d_a_pressed:
            keyboard.release(Key.down)
            print("DOWN RELEASED")
            d_a_pressed = False

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv.imshow("Video", img)


    if cv.waitKey(20) & 0xFF == ord('l'):
        break

cap.release()
cv.destroyAllWindows()
