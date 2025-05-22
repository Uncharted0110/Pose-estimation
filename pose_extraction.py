import cv2
import csv
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open video
video_path = "./pushup/push-up.mp4"  # <--- Change this
cap = cv2.VideoCapture(video_path)

# Output CSV
csv_file = open("pushup_front.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

# Header
landmark_names = [f"{joint}_{axis}" for joint in range(33) for axis in ["x", "y", "z", "v"]]
csv_writer.writerow(["frame", *landmark_names, "label"])

frame_num = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Save to CSV
        row = [frame_num]
        for landmark in results.pose_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        row.append("pushup")
        csv_writer.writerow(row)

        # Draw landmarks on frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the video with landmarks
    cv2.imshow('Pose Detection', frame)

    # Press Esc to quit early
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_num += 1

# Cleanup
csv_file.close()
cap.release()
cv2.destroyAllWindows()
pose.close()
print("✅ Done. Landmarks saved and video processed.")
