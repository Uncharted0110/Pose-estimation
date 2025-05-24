import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# Load model, scaler, and ideal pushup pose
model = joblib.load("pushup_detector.pkl")
scaler = joblib.load("scaler.pkl")

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Prediction history and counters
prediction_history = deque(maxlen=50)
rep_count = 0
stage = "up"
label = "Detecting..."

# Multi-Level Feedback Tracking System
class MultiLevelFeedbackTracker:
    def __init__(self):
        # Define the levels in order of progression
        self.levels = {
            1: {'name': 'BODY_ALIGNMENT', 'passed': False, 'message': 'Keep your body straight'},
            2: {'name': 'DESCENT_STARTED', 'passed': False, 'message': 'Start lowering your body'},
            3: {'name': 'DEPTH_REACHED', 'passed': False, 'message': 'Go lower to complete the pushup'},
            4: {'name': 'ASCENT_COMPLETE', 'passed': False, 'message': 'Push back up to complete the rep'}
        }
        
        self.current_level = 1
        self.deepest_level_reached = 1
        self.feedback_message = "Position yourself for pushup"
        self.rep_in_progress = False
        self.last_nose_y = None
        self.descent_started_y = None
        self.body_straight_frames = 0
        self.body_straight_threshold = 10  # Need 10 consecutive frames of straight body
        
        # Thresholds
        self.straight_body_angle_threshold = 160
        self.descent_threshold = 0.05  # How much nose needs to move down to start descent
        self.depth_threshold = 0.6  # Nose y-position for adequate depth
        self.ascent_threshold = 0.5  # Nose y-position for completed ascent
        
    def reset_rep(self):
        """Reset all levels for a new rep"""
        for level in self.levels.values():
            level['passed'] = False
        self.current_level = 1
        self.deepest_level_reached = 1
        self.rep_in_progress = False
        self.descent_started_y = None
        self.body_straight_frames = 0
        
    def check_body_alignment(self, landmarks):
        """Level 1: Check if body is straight"""
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        mid_shoulder = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
        mid_hip = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
        mid_knee = ((lk.x + rk.x) / 2, (lk.y + rk.y) / 2)
        
        back_angle = self.angle_between_points(mid_shoulder, mid_hip, mid_knee)
        
        if back_angle >= self.straight_body_angle_threshold:
            self.body_straight_frames += 1
            if self.body_straight_frames >= self.body_straight_threshold:
                return True
        else:
            self.body_straight_frames = 0
            
        return False
    
    def check_descent_started(self, nose_y):
        """Level 2: Check if descent has started"""
        if self.last_nose_y is not None:
            if self.descent_started_y is None:
                self.descent_started_y = self.last_nose_y
            
            descent_amount = nose_y - self.descent_started_y
            if descent_amount >= self.descent_threshold:
                return True
        return False
    
    def check_depth_reached(self, nose_y):
        """Level 3: Check if adequate depth is reached"""
        return nose_y >= self.depth_threshold
    
    def check_ascent_complete(self, nose_y):
        """Level 4: Check if ascent is complete"""
        return nose_y <= self.ascent_threshold
    
    def angle_between_points(self, p1, p2, p3):
        """Calculate angle (in degrees) at p2 formed by points p1-p2-p3"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        ba = a - b
        bc = c - b

        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def update_levels(self, landmarks, nose_y, is_pushup_detected):
        """Update the progression through levels"""
        if not is_pushup_detected:
            if self.rep_in_progress:
                # If we were in a rep but no longer detecting pushup pose, reset
                self.reset_rep()
            self.feedback_message = "Position yourself for pushup"
            return
        
        # Start tracking a new rep if not in progress and body is aligned
        if not self.rep_in_progress:
            if self.check_body_alignment(landmarks):
                self.rep_in_progress = True
                self.levels[1]['passed'] = True
                self.current_level = 2
                self.deepest_level_reached = max(self.deepest_level_reached, 1)
        
        if self.rep_in_progress:
            # Level 1: Body Alignment (already passed to start rep)
            
            # Level 2: Descent Started
            if self.current_level == 2 and not self.levels[2]['passed']:
                if self.check_descent_started(nose_y):
                    self.levels[2]['passed'] = True
                    self.current_level = 3
                    self.deepest_level_reached = max(self.deepest_level_reached, 2)
            
            # Level 3: Depth Reached
            if self.current_level == 3 and not self.levels[3]['passed']:
                if self.check_depth_reached(nose_y):
                    self.levels[3]['passed'] = True
                    self.current_level = 4
                    self.deepest_level_reached = max(self.deepest_level_reached, 3)
            
            # Level 4: Ascent Complete
            if self.current_level == 4 and not self.levels[4]['passed']:
                if self.check_ascent_complete(nose_y):
                    self.levels[4]['passed'] = True
                    self.deepest_level_reached = max(self.deepest_level_reached, 4)
                    # Rep completed successfully!
                    self.feedback_message = "✅ Excellent Form! Rep Completed!"
                    return True  # Signal rep completion
        
        # Update feedback message based on deepest level reached
        self.update_feedback_message()
        self.last_nose_y = nose_y
        return False
    
    def update_feedback_message(self):
        """Update feedback message based on current progress"""
        if self.deepest_level_reached == 1 and self.levels[1]['passed']:
            self.feedback_message = "✅ Body aligned! Start lowering down"
        elif self.deepest_level_reached == 2 and self.levels[2]['passed']:
            self.feedback_message = "✅ Descent started! Go lower"
        elif self.deepest_level_reached == 3 and self.levels[3]['passed']:
            self.feedback_message = "✅ Good depth! Push back up"
        else:
            # Provide feedback for the current level that needs to be achieved
            if not self.levels[1]['passed']:
                self.feedback_message = "⚠️ " + self.levels[1]['message']
            elif not self.levels[2]['passed']:
                self.feedback_message = "⚠️ " + self.levels[2]['message']
            elif not self.levels[3]['passed']:
                self.feedback_message = "⚠️ " + self.levels[3]['message']
            elif not self.levels[4]['passed']:
                self.feedback_message = "⚠️ " + self.levels[4]['message']
    
    def get_debug_info(self):
        """Get debugging information for display"""
        debug_info = []
        debug_info.append(f"Rep in Progress: {self.rep_in_progress}")
        debug_info.append(f"Current Level: {self.current_level}")
        debug_info.append(f"Deepest Level: {self.deepest_level_reached}")
        debug_info.append(f"Body Straight Frames: {self.body_straight_frames}/{self.body_straight_threshold}")
        
        for level_num, level_data in self.levels.items():
            status = "✅" if level_data['passed'] else "❌"
            debug_info.append(f"L{level_num} {level_data['name']}: {status}")
            
        return debug_info
    
    def get_feedback(self):
        """Get current feedback message"""
        return self.feedback_message

# Create feedback tracker instance
feedback_tracker = MultiLevelFeedbackTracker()

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
        data = []

        for lm in landmarks:
            data.extend([lm.x, lm.y, lm.z, lm.visibility])

        # Predict label
        x = scaler.transform([data])
        pred = model.predict(x)[0]
        prediction_history.append(pred)

        # Voting
        if len(prediction_history) == prediction_history.maxlen:
            vote = sum(prediction_history)
            label = "Pushup" if vote > len(prediction_history) * 0.6 else "Not Pushup"

        # Get nose position
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        
        # Update multi-level feedback system
        rep_completed = feedback_tracker.update_levels(landmarks, nose.y, label == "Pushup")
        
        if rep_completed:
            rep_count += 1
            feedback_tracker.reset_rep()

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    else:
        feedback_tracker.feedback_message = "Position yourself in view"

    # Display main info
    cv2.putText(frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f'Reps: {rep_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    
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
    
    y_offset = 110
    for line in feedback_lines:
        color = (0,0,255) if "⚠️" in line else (0,255,0)
        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30

    # Display debug information
    debug_info = feedback_tracker.get_debug_info()
    debug_y_offset = 200
    cv2.putText(frame, "=== DEBUG INFO ===", (10, debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    debug_y_offset += 25
    
    for debug_line in debug_info:
        cv2.putText(frame, debug_line, (10, debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        debug_y_offset += 20

    # Show frame
    cv2.imshow("Pushup Detector", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()