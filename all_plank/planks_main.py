import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Feedback tracker for Plank
class PlankFeedbackTracker:
    def __init__(self):
        self.stable_frames = 0
        self.stable_threshold = 20  # Number of frames to confirm plank
        self.position_lost_frames = 0
        self.max_position_lost_frames = 30
        self.feedback_message = "Get into plank position"
        self.plank_held = False
        self.deepest_level_reached = 0
        self.current_hip_height = 0
        self.current_shoulder_hip_ankle_angle = 0
        self.min_angle = 150  # Minimum angle for straight body
        self.max_angle = 180  # Maximum angle for straight body
        self.hip_height_min = 0.35  # Min y for hips (not too high)
        self.hip_height_max = 0.65  # Max y for hips (not too low)

    def reset(self):
        self.stable_frames = 0
        self.position_lost_frames = 0
        self.plank_held = False
        self.feedback_message = "Get into plank position"

    def check_body_alignment(self, landmarks):
        # Get key points
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Midpoints
        mid_shoulder = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        mid_hip = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        mid_ankle = ((left_ankle.x + right_ankle.x) / 2, (left_ankle.y + right_ankle.y) / 2)

        # Angle at hip (shoulder-hip-ankle)
        angle = self.angle_between_points(mid_shoulder, mid_hip, mid_ankle)
        self.current_shoulder_hip_ankle_angle = round(angle, 1)
        return self.min_angle < angle < self.max_angle

    def check_hip_height(self, landmarks):
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        mid_hip_y = (left_hip.y + right_hip.y) / 2
        self.current_hip_height = round(mid_hip_y, 3)
        return self.hip_height_min < mid_hip_y < self.hip_height_max

    def angle_between_points(self, p1, p2, p3):
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def update(self, landmarks):
        aligned = self.check_body_alignment(landmarks)
        hip_ok = self.check_hip_height(landmarks)
        if aligned and hip_ok:
            self.stable_frames += 1
            self.position_lost_frames = 0
            if self.stable_frames >= self.stable_threshold:
                self.plank_held = True
                self.feedback_message = "✅ Good plank! Hold steady."
        else:
            if self.plank_held:
                self.position_lost_frames += 1
                if self.position_lost_frames > self.max_position_lost_frames:
                    self.reset()
                    self.feedback_message = "Position lost! Get back into plank."
            else:
                self.stable_frames = 0
                if not aligned:
                    self.feedback_message = f"⚠️ Keep your body straight! Angle: {self.current_shoulder_hip_ankle_angle} deg"
                elif not hip_ok:
                    self.feedback_message = f"⚠️ Adjust hip height! y: {self.current_hip_height}"
        return self.plank_held

    def draw_visualization(self, frame, landmarks):
        h, w, _ = frame.shape
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        mid_shoulder = (int((left_shoulder.x + right_shoulder.x) / 2 * w), int((left_shoulder.y + right_shoulder.y) / 2 * h))
        mid_hip = (int((left_hip.x + right_hip.x) / 2 * w), int((left_hip.y + right_hip.y) / 2 * h))
        mid_ankle = (int((left_ankle.x + right_ankle.x) / 2 * w), int((left_ankle.y + right_ankle.y) / 2 * h))
        # Draw lines
        cv2.line(frame, mid_shoulder, mid_hip, (255, 0, 255), 3)
        cv2.line(frame, mid_hip, mid_ankle, (0, 255, 255), 3)
        # Draw angle arc at hip
        if self.current_shoulder_hip_ankle_angle > 10:
            cv2.ellipse(frame, mid_hip, (60, 60), 0, 0, int(self.current_shoulder_hip_ankle_angle), (255, 255, 255), 2)
        # Add angle text
        cv2.putText(frame, f"Body Angle: {self.current_shoulder_hip_ankle_angle} deg", (mid_hip[0] + 70, mid_hip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def get_feedback(self):
        return self.feedback_message

    def get_debug_info(self):
        debug_info = []
        debug_info.append(f"Stable Frames: {self.stable_frames}/{self.stable_threshold}")
        debug_info.append(f"Position Lost Frames: {self.position_lost_frames}")
        debug_info.append(f"Hip Height: {self.current_hip_height}")
        debug_info.append(f"Body Angle: {self.current_shoulder_hip_ankle_angle} deg")
        debug_info.append(f"Plank Held: {self.plank_held}")
        return debug_info

# Main loop
feedback_tracker = PlankFeedbackTracker()
# For webcam use:
cap = cv2.VideoCapture(0)
# If you want to use a video file instead, comment the above line and uncomment the following:
# cap = cv2.VideoCapture(r"e:/Mini_project_python/Pose-estimation/all_plank/plank_videos/plank-5.mp4")
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()
# Get FPS for time calculation
fps = cap.get(cv2.CAP_PROP_FPS)
total_plank_frames = 0
plank_time_sec = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.flip(frame, 1)  # Commented out for video file
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        plank_held = feedback_tracker.update(landmarks)
        if plank_held:
            total_plank_frames += 1  # Accumulate total frames
        plank_time_sec = total_plank_frames / fps
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        feedback_tracker.draw_visualization(frame, landmarks)
        label = "Plank Position"
    else:
        feedback_tracker.feedback_message = "Position yourself in view for plank"
        label = "No pose detected"
    # Display info
    cv2.putText(frame, f'Exercise: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    # Display plank time
    cv2.putText(frame, f'Plank Time: {plank_time_sec:.1f} sec', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)
    # Display feedback
    feedback = feedback_tracker.get_feedback()
    feedback_lines = []
    if len(feedback) > 50:
        words = feedback.split()
        current_line = words[0]
        for word in words[1:]:
            if len(current_line + " " + word) <= 50:
                current_line += " " + word
            else:
                feedback_lines.append(current_line)
                current_line = word
        feedback_lines.append(current_line)
    else:
        feedback_lines.append(feedback)
    y_offset = 70
    for line in feedback_lines:
        color = (0,0,255) if "⚠️" in line else (0,255,0)
        # Move feedback text a bit lower (e.g., y_offset + 40)
        cv2.putText(frame, line, (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25
    # Display debug info
    debug_info = feedback_tracker.get_debug_info()
    debug_y_offset = 150
    cv2.putText(frame, "=== DEBUG INFO ===", (10, debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
    debug_y_offset += 20
    for debug_line in debug_info:
        cv2.putText(frame, debug_line, (10, debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        debug_y_offset += 18
    cv2.imshow("Plank Detector", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()