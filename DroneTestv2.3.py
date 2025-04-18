import cv2 as cv
import mediapipe as mp
import hand_track_module as htmod
import time
import numpy as np
import vgamepad as vg

# Constants and initializations
wCam, hCam = 640, 480
alpha_throttle = 0.15  # Smoothing factor
alpha_pitch = 0.4  # Smoothed pitch (now controlled by index and ring fingers)
alpha_yaw = 0.1  # Smoothed yaw (now controlled by hand tilt)
detector = htmod.handDetector(detectionCon=0.7)
gamepad = vg.VX360Gamepad()

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

prevTime = 0
smooth_throttle = smooth_pitch = smooth_yaw = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        #print("Failed to capture frame. Exiting...")
        break

    # Hand detection
    frame = detector.findHands(frame, draw=False)
    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) == 0:
        #print("No hand detected. Resetting gamepad.")
        gamepad.reset()
        gamepad.right_joystick(0, 0)
        gamepad.left_joystick(0, 0)
        gamepad.update()
        continue

    # Extract finger landmarks
    x0, y0 = lmList[0][1], lmList[0][2]    # Wrist
    x2, y2 = lmList[8][1], lmList[8][2]    # Index finger tip
    x4, y4 = lmList[16][1], lmList[16][2]  # Ring finger tip
    x5, y5 = lmList[5][1], lmList[5][2]    # Base of index finger (used for tilt)
    x9, y9 = lmList[9][1], lmList[9][2]    # Base of middle finger (used for tilt reference)

    # Throttle control (left joystick Y-axis)
    throttle_control = abs(y2 - y0)
    throttle_min, throttle_max = 170, 450
    throttle_control = np.clip(throttle_control, throttle_min, throttle_max)
    smooth_throttle = smooth_throttle * (1 - alpha_throttle) + throttle_control * alpha_throttle

    # Pitch control (right joystick Y-axis, controlled by index and ring fingers)
    pitch_value = y4 - y2
    pitch_min, pitch_max = -210, 210
    pitch_value = np.clip(pitch_value, pitch_min, pitch_max)
    smooth_pitch = smooth_pitch * (1 - alpha_pitch) + pitch_value * alpha_pitch

    # Yaw control (left joystick X-axis, controlled by hand tilt)
    yaw_value = x5 - x9  # Difference in x-coordinates between index base and middle base
    yaw_min, yaw_max = -150, 150
    yaw_value = np.clip(yaw_value, yaw_min, yaw_max)
    smooth_yaw = smooth_yaw * (1 - alpha_yaw) + yaw_value * alpha_yaw

    # Map values to joystick ranges
    pitch_right_stick_y = int(np.interp(smooth_pitch, [pitch_min, pitch_max], [-32767, 32767]))  # Pitch control
    yaw_left_stick_x = int(np.interp(smooth_yaw, [yaw_min, yaw_max], [-32767, 32767]))  # Yaw control
    throttle_left_stick_y = int(np.interp(smooth_throttle, [throttle_min, throttle_max], [32767, -32767]))  # Throttle

    # Update gamepad
    gamepad.right_joystick(0, pitch_right_stick_y)  # Only pitch on right stick
    gamepad.left_joystick(yaw_left_stick_x, throttle_left_stick_y)  # Yaw and throttle on left stick
    gamepad.update()

    # FPS calculation
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv.putText(frame, f"FPS: {int(fps)}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Debug print
    #print(f"Throttle: {smooth_throttle} | Pitch: {smooth_pitch} | Yaw: {smooth_yaw} | 
          #Right Joystick Y (Pitch): {pitch_right_stick_y} | Left Joystick X (Yaw): {yaw_left_stick_x} | Left Joystick Y (Throttle): {throttle_left_stick_y}")

    # Show frame
    frame = cv.resize(frame, (640, 480))
    cv.imshow('frame', frame)

    # Break on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program.")
        break

cap.release()
cv.destroyAllWindows()