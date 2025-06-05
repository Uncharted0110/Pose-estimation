import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Prediction history and counters
rep_count = 0
label = "Detecting..."

# Multi-Level Feedback Tracking System for Squats
class SquatFeedbackTracker:
    def __init__(self):
        self.levels = {
            1: {'name': 'STANDING', 'passed': False, 'message': 'Stand upright, feet shoulder-width apart'},
            2: {'name': 'DESCENT', 'passed': False, 'message': 'Lower hips, bend knees to squat'},
            3: {'name': 'BOTTOM', 'passed': False, 'message': 'Thighs parallel to ground'},
            4: {'name': 'ASCENT', 'passed': False, 'message': 'Return to standing position'}
        }
        self.current_level = 1
        self.deepest_level_reached = 1
        self.feedback_message = "Get into squat position"
        self.squat_in_progress = False
        self.bottom_reached = False
        self.standing_threshold = 165  # degrees
        self.squat_threshold = 100     # degrees
        self.bottom_threshold = 90     # degrees
        self.knee_angle = 180
        self.hip_angle = 180
        self.position_lost_frames = 0
        self.max_position_lost_frames = 30

    def reset(self):
        for level in self.levels.values():
            level['passed'] = False
        self.current_level = 1
        self.deepest_level_reached = 1
        self.squat_in_progress = False
        self.bottom_reached = False
        self.position_lost_frames = 0

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def update(self, landmarks):
        rep_completed = False
        # Get relevant points
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # Calculate knee and hip angles (average both sides)
        left_knee_angle = self.calculate_angle(
            (left_hip.x, left_hip.y), (left_knee.x, left_knee.y), (left_ankle.x, left_ankle.y))
        right_knee_angle = self.calculate_angle(
            (right_hip.x, right_hip.y), (right_knee.x, right_knee.y), (right_ankle.x, right_ankle.y))
        self.knee_angle = (left_knee_angle + right_knee_angle) / 2

        left_hip_angle = self.calculate_angle(
            (left_shoulder.x, left_shoulder.y), (left_hip.x, left_hip.y), (left_knee.x, left_knee.y))
        right_hip_angle = self.calculate_angle(
            (right_shoulder.x, right_shoulder.y), (right_hip.x, right_hip.y), (right_knee.x, right_knee.y))
        self.hip_angle = (left_hip_angle + right_hip_angle) / 2

        # Level 1: Standing
        if self.knee_angle > self.standing_threshold and self.hip_angle > self.standing_threshold:
            self.levels[1]['passed'] = True
            self.levels[2]['passed'] = False
            self.levels[3]['passed'] = False
            self.levels[4]['passed'] = False
            self.squat_in_progress = False
            self.bottom_reached = False
            self.current_level = 1
            self.deepest_level_reached = max(self.deepest_level_reached, 1)
            self.feedback_message = "Stand upright, feet shoulder-width apart"
        # Level 2: Descent
        elif self.knee_angle < self.standing_threshold and self.knee_angle > self.squat_threshold:
            self.levels[2]['passed'] = True
            self.current_level = 2
            self.deepest_level_reached = max(self.deepest_level_reached, 2)
            self.squat_in_progress = True
            self.feedback_message = "Lower hips, bend knees to squat"
        # Level 3: Bottom
        elif self.knee_angle <= self.squat_threshold and self.knee_angle > self.bottom_threshold:
            self.levels[3]['passed'] = True
            self.current_level = 3
            self.deepest_level_reached = max(self.deepest_level_reached, 3)
            self.bottom_reached = True
            self.feedback_message = "Go a bit lower for full squat!"
        elif self.knee_angle <= self.bottom_threshold:
            self.levels[3]['passed'] = True
            self.current_level = 3
            self.deepest_level_reached = max(self.deepest_level_reached, 3)
            self.bottom_reached = True
            self.feedback_message = "Thighs parallel to ground!"
        # Level 4: Ascent
        elif self.squat_in_progress and self.bottom_reached and self.knee_angle > self.standing_threshold:
            self.levels[4]['passed'] = True
            self.current_level = 4
            self.deepest_level_reached = max(self.deepest_level_reached, 4)
            self.squat_in_progress = False
            self.bottom_reached = False
            rep_completed = True
            self.feedback_message = "Good squat! Stand up fully."
        else:
            self.feedback_message = "Maintain squat form."
        return rep_completed

    def draw_visualization(self, frame, landmarks):
        h, w, _ = frame.shape
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        # Draw left leg
        cv2.line(frame, (int(left_hip.x*w), int(left_hip.y*h)), (int(left_knee.x*w), int(left_knee.y*h)), (255,0,0), 3)
        cv2.line(frame, (int(left_knee.x*w), int(left_knee.y*h)), (int(left_ankle.x*w), int(left_ankle.y*h)), (255,0,0), 3)
        # Draw right leg
        cv2.line(frame, (int(right_hip.x*w), int(right_hip.y*h)), (int(right_knee.x*w), int(right_knee.y*h)), (0,255,0), 3)
        cv2.line(frame, (int(right_knee.x*w), int(right_knee.y*h)), (int(right_ankle.x*w), int(right_ankle.y*h)), (0,255,0), 3)
        # Draw knee angle
        mid_knee = (int((left_knee.x+right_knee.x)/2*w), int((left_knee.y+right_knee.y)/2*h))
        cv2.putText(frame, f"Knee Angle: {int(self.knee_angle)}", (mid_knee[0]-40, mid_knee[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        # Draw hip angle
        mid_hip = (int((left_hip.x+right_hip.x)/2*w), int((left_hip.y+right_hip.y)/2*h))
        cv2.putText(frame, f"Hip Angle: {int(self.hip_angle)}", (mid_hip[0]-40, mid_hip[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    def get_feedback(self):
        return self.feedback_message

    def get_debug_info(self):
        debug_info = []
        debug_info.append(f"Knee Angle: {self.knee_angle:.1f}")
        debug_info.append(f"Hip Angle: {self.hip_angle:.1f}")
        for level_num, level_data in self.levels.items():
            status = "✅" if level_data['passed'] else "❌"
            debug_info.append(f"L{level_num} {level_data['name']}: {status}")
        return debug_info

# Create feedback tracker instance
feedback_tracker = SquatFeedbackTracker()

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        label = "Squat Position"
        rep_completed = feedback_tracker.update(landmarks)
        if rep_completed:
            rep_count += 1
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        feedback_tracker.draw_visualization(frame, landmarks)
    else:
        feedback_tracker.feedback_message = "Position yourself in view for squats"
        label = "No pose detected"
    cv2.putText(frame, f'Exercise: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f'Reps: {rep_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
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
    y_offset = 110
    for line in feedback_lines:
        color = (0,0,255) if "squat" in line.lower() or "maintain" in line.lower() else (0,255,0)
        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25
    debug_info = feedback_tracker.get_debug_info()
    debug_y_offset = 200
    cv2.putText(frame, "=== DEBUG INFO ===", (10, debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
    debug_y_offset += 20
    for debug_line in debug_info:
        cv2.putText(frame, debug_line, (10, debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        debug_y_offset += 18
    cv2.imshow("Squat Detector", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()