import cv2
import mediapipe as mp
import pandas as pd
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

base_dir = "non_pushup"
num_videos = 9

all_data = []

for i in range(1, num_videos + 1):
    video_path = os.path.join(base_dir, f"non_pushup-{i}.mp4")
    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        continue

    print(f"Processing {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            row = [frame_num]
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            row.append(0)
            all_data.append(row)
        else:
            # No landmarks
            row = [frame_num] + [float('nan')] * (33 * 4) + [0]
            all_data.append(row)

        # Show frame with landmarks
        cv2.imshow('Pushup Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

pose.close()
cv2.destroyAllWindows()

columns = ['frame']
for i in range(33):
    columns.extend([f"{i}_x", f"{i}_y", f"{i}_z", f"{i}_v"])
columns.append('label')

df = pd.DataFrame(all_data, columns=columns)
df.to_csv("non_pushup_landmarks.csv", index=False)

print("Extraction complete! Data saved to non_pushup_landmarks.csv")