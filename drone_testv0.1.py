import cv2 as cv
import mediapipe as mp
import HandTrackingModule as htmod
import time
import numpy as np
import vgamepad as vg

wCam, hCam = 640, 480

detector = htmod.handDetector(detectionCon=0.7)
gamepad = vg.VX360Gamepad()

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

prevTime = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) == 0:
        print("No hand detected. Resetting gamepad.")
        gamepad.reset()
        gamepad.right_joystick(0, 0)
        gamepad.left_joystick(0, 0)
        gamepad.update()
        continue

    # Finger landmarks
    x3, y3 = lmList[12][1], lmList[12][2]  # Middle finger (for yaw)
    x0, y0 = lmList[0][1], lmList[0][2]    # Wrist
    x2, y2 = lmList[8][1], lmList[8][2]    # Index finger
    x4, y4 = lmList[16][1], lmList[16][2]  # Ring finger

    # Debugging prints
    print(f"Middle Finger: ({x3}, {y3}), Wrist: ({x0}, {y0})")
    print(f"Index Finger: ({x2}, {y2}), Ring Finger: ({x4}, {y4})")

    cv.circle(frame, (x3, y3), 15, (255, 0, 255), -1)
    cv.circle(frame, (x0, y0), 15, (255, 0, 255), -1)
    cv.line(frame, (x0, y0), (x3, y3), (255, 0, 0), 3)

    # Throttle control (left joystick Y-axis)
    throttle_control = abs(y3 - y0)
    throttle_min, throttle_max = 100, 500
    throttle_control = np.clip(throttle_control, throttle_min, throttle_max)
    print(f"Throttle Control: {throttle_control} (Clipped to [{throttle_min}, {throttle_max}])")

    # Roll control (now mapped to right joystick X-axis)
    if x4 != x2:
        slope = (y4 - y2) / (x4 - x2)
        roll_inclination = np.degrees(np.arctan(slope))
        print(f"Roll Slope: {slope}, Roll Inclination: {roll_inclination} degrees")
    else:
        roll_inclination = 90
        print("Vertical line detected. Roll inclination set to 90 degrees.")

    cv.line(frame, (x2, y2), (x4, y4), (255, 0, 0), 3)

    # Yaw control (right joystick Y-axis)
    yaw = x3 - x0
    print(f"yaw:{yaw}")
    # print(f"Yaw Slope: {yaw_slope}, Yaw Inclination: {yaw_inclination} degrees")

    # Map values to joystick ranges
    roll_right_stick_x = int(np.interp(roll_inclination, [-45, 45], [-32767, 32767]))  # Roll control
    yaw_left_stick_x = int(np.interp(yaw, [-440, 440], [32767, -32767]))  # Yaw control
    
    throttle_left_stick_y = int(np.interp(throttle_control, [throttle_min, throttle_max], [32767, -32767]))  # Throttle control

    print(f"Right Joystick X (Roll): {roll_right_stick_x}")
    print(f"left Joystick X (Yaw): {yaw_left_stick_x}")
    print(f"Left Joystick Y (Throttle): {throttle_left_stick_y}")

    # Update gamepad
    gamepad.right_joystick(roll_right_stick_x, 0)  
    gamepad.left_joystick(yaw_left_stick_x, throttle_left_stick_y)              
    gamepad.update()
    print("Gamepad updated with new joystick values.")

    # FPS calculation
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv.putText(frame, f"FPS: {int(fps)}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    print(f"FPS: {int(fps)}")

    frame = cv.resize(frame, (640, 480))
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program.")
        break

cap.release()
cv.destroyAllWindows()
