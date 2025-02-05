import cv2 as cv
import mediapipe as mp
import handTrackingModule as htmod
import time
import numpy as np
import vgamepad as vg

wCam, hCam = 640, 480

# Hand detector and virtual gamepad setup
detector = htmod.handDetector(detectionCon=0.7)
gamepad = vg.VX360Gamepad()

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

prevTime = 0
alpha = 0.2  # Smoothing factor

# Initial values for smoothing
smooth_throttle = 0
smooth_turn = 0
smooth_yaw = 0
smooth_pitch = 0

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

    # Extract finger landmarks
    x3, y3 = lmList[12][1], lmList[12][2]  # Middle finger (for yaw)
    x0, y0 = lmList[0][1], lmList[0][2]    # Wrist
    x2, y2 = lmList[8][1], lmList[8][2]    # Index finger
    x4, y4 = lmList[16][1], lmList[16][2]  # Ring finger

    # Throttle (Left Stick Y - Up/Down)
    throttle_control = abs(y3 - y0)
    throttle_min, throttle_max = 100, 400
    throttle = np.clip(throttle_control, throttle_min, throttle_max)
    smooth_throttle = smooth_throttle * (1 - alpha) + throttle * alpha
    throttle_value = int(np.interp(smooth_throttle, [throttle_min, throttle_max], [32767, -32767]))

    # Turn (Left Stick X - Left/Right)
    turn_control = x3 - x0
    smooth_turn = smooth_turn * (1 - alpha) + turn_control * alpha
    turn_value = int(np.interp(smooth_turn, [-300, 300], [-32767, 32767]))

    # Yaw (Right Stick X - Left/Right, Limited Range)
    yaw_control = x4 - x2
    smooth_yaw = smooth_yaw * (1 - alpha) + yaw_control * alpha
    yaw_value = int(np.interp(smooth_yaw, [-200, 200], [-16000, 16000]))  # Limited to avoid flipping

    # Pitch (Right Stick Y - Up/Down)
    pitch_control = y4 - y2
    smooth_pitch = smooth_pitch * (1 - alpha) + pitch_control * alpha
    pitch_value = int(np.interp(smooth_pitch, [-200, 200], [32767, -32767]))

    # Update gamepad
    gamepad.left_joystick(turn_value, throttle_value)  # Turn + Throttle
    gamepad.right_joystick(yaw_value, pitch_value)  # Yaw + Pitch
    gamepad.update()

    # Debug print
    print(f"Throttle: {smooth_throttle:.2f} (LS Y: {throttle_value}), Turn: {smooth_turn:.2f} (LS X: {turn_value}), Yaw: {smooth_yaw:.2f} (RS X: {yaw_value}), Pitch: {smooth_pitch:.2f} (RS Y: {pitch_value})")

    # FPS calculation
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv.putText(frame, f"FPS: {int(fps)}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program.")
        break

cap.release()
cv.destroyAllWindows()