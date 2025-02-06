import cv2 as cv
import mediapipe as mp
import HandTrackingModule as htmod
import time
import numpy as np
import vgamepad as vg

# Constants and initializations
wCam, hCam = 640, 480
alpha_throttle = 0.15  # Smoothing factor
alpha_roll = 0.4
alpha_yaw = 0.1
detector = htmod.handDetector(detectionCon=0.7)
gamepad = vg.VX360Gamepad()

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

prevTime = 0
smooth_throttle = smooth_roll = smooth_yaw = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Hand detection
    frame = detector.findHands(frame, draw=False)
    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) == 0:
        print("No hand detected. Resetting gamepad.")
        gamepad.reset()
        gamepad.right_joystick(0, 0)
        gamepad.left_joystick(0, 0)
        gamepad.update()
        continue

    # Extract finger landmarks
    x3, y3 = lmList[12][1], lmList[12][2]  # Middle finger (for yaw)
    x0, y0 = lmList[0][1], lmList[0][2]    # Wrist
    x2, y2 = lmList[8][1], lmList[8][2]    # Index finger
    x4, y4 = lmList[16][1], lmList[16][2]  # Ring finger

    cv.line(frame, (x0, y0), (x3, y3), (255, 0, 0), 3)
    cv.line(frame, (x2, y2), (x4, y4), (255, 0, 0), 3)

    # Throttle control (left joystick Y-axis)
    throttle_control = abs(y3 - y0)
    throttle_min, throttle_max = 170, 450
    throttle_control = np.clip(throttle_control, throttle_min, throttle_max)
    
    smooth_throttle = smooth_throttle * (1 - alpha_throttle) + throttle_control * alpha_throttle

    # Roll control (now mapped to right joystick X-axis)
    roll_value = y4 - y2
    roll_min, roll_max = -210, 210
    roll_value = np.clip(roll_value, roll_min, roll_max)
    smooth_roll = smooth_roll * (1 - alpha_roll) + roll_value * alpha_roll

    # Yaw control (right joystick Y-axis)
    yaw = x3 - x0
    yaw_min, yaw_max = -250, 250
    yaw = np.clip(yaw, yaw_min, yaw_max)
    smooth_yaw = smooth_yaw * (1 - alpha_yaw) + yaw * alpha_yaw

    # Map values to joystick ranges
    roll_right_stick_x = int(np.interp(smooth_roll, [roll_min, roll_max], [-32767, 32767]))  # Roll control
    yaw_left_stick_x = int(np.interp(smooth_yaw, [yaw_min, yaw_max], [32767, -32767]))  # Yaw control
    throttle_left_stick_y = int(np.interp(smooth_throttle, [throttle_min, throttle_max], [32767, -32767]))  # Throttle control

    # Update gamepad
    gamepad.right_joystick(roll_right_stick_x, 0)  
    gamepad.left_joystick(yaw_left_stick_x, throttle_left_stick_y)              
    gamepad.update()

    # FPS calculation
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv.putText(frame, f"FPS: {int(fps)}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Print all values in a single line
    print(f"Throttle (Before Smooth): {throttle_control} | Throttle (After Smooth): {smooth_throttle} | "
          f"Roll (Before Smooth): {roll_value} | Roll (After Smooth): {smooth_roll} | "
          f"Yaw (Before Smooth): {yaw} | Yaw (After Smooth): {smooth_yaw} | "
          f"Right Joystick X (Roll): {roll_right_stick_x} | Left Joystick X (Yaw): {yaw_left_stick_x} | "
          f"Left Joystick Y (Throttle): {throttle_left_stick_y}")

    # Show frame
    frame = cv.resize(frame, (640, 480))
    cv.imshow('frame', frame)

    # Break on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program.")
        break

cap.release()
cv.destroyAllWindows()
