# AURA-OS — Layer 5: Perception Layer
# gesture.py — Week 1+3: Camera + Gesture Classification
# Author: Samala Shashanth | Project: AURA-OS

import cv2  # type: ignore
import mediapipe as mp  # type: ignore

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def count_fingers(landmarks, hand_label="Right"):
    tips    = [8, 12, 16, 20]
    knuckle = [6, 10, 14, 18]
    count: int = 0

    # Thumb — compare x (direction depends on handedness)
    if hand_label == "Right":
        if landmarks[4].x < landmarks[3].x:
            count += 1
    else:  # Left hand
        if landmarks[4].x > landmarks[3].x:
            count += 1

    # Four fingers
    count = count + sum(
        1 for tip_id, mcp_id in zip(tips, knuckle)
        if landmarks[tip_id].y < landmarks[mcp_id].y
    )

    return count

def classify_gesture(landmarks, hand_label="Right"):
    fingers = count_fingers(landmarks, hand_label)
    if fingers == 0: return "FIST"
    if fingers == 1: return "POINT"
    if fingers == 2: return "PEACE"
    if fingers == 3: return "THREE"
    if fingers == 5: return "OPEN_PALM"
    return f"{fingers}_FINGERS"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not found. Try VideoCapture(1)")
    exit()

print("AURA-OS Layer 5 — Gesture classifier running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Extract handedness for correct thumb detection
            hand_label = "Right"
            if result.multi_handedness:
                hand_label = result.multi_handedness[idx].classification[0].label

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = classify_gesture(hand_landmarks.landmark, hand_label)

            # Display gesture name on screen
            cv2.putText(frame, gesture, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 100), 3)

            print(f"Gesture: {gesture}")

    cv2.imshow("AURA-OS | Layer 5 - Gesture", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Layer 5 Gesture Classification complete.")