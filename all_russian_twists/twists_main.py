import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# Load model, scaler (you'll need to train these for Russian twists)
# model = joblib.load("russian_twists_detector.pkl")
# scaler = joblib.load("scaler.pkl")

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Prediction history and counters
prediction_history = deque(maxlen=30)
rep_count = 0
current_side = "center"
last_side = "center"
label = "Detecting..."

# Multi-Level Feedback Tracking System for Russian Twists
class RussianTwistsFeedbackTracker:
    def __init__(self):
        # Define the levels in order of progression
        self.levels = {
            1: {'name': 'SEATED_POSITION', 'passed': False, 'message': 'Sit down with knees bent'},
            2: {'name': 'LEGS_ELEVATED', 'passed': False, 'message': 'Lift your legs off the ground'},
            3: {'name': 'TORSO_LEAN', 'passed': False, 'message': 'Lean back to create V-shape with torso and thighs'},
            4: {'name': 'READY_TO_TWIST', 'passed': False, 'message': 'Hold position, ready to twist'},
            5: {'name': 'TWISTING_MOTION', 'passed': False, 'message': 'Touch elbow to opposite knee'}
        }
        
        self.current_level = 1
        self.deepest_level_reached = 1
        self.feedback_message = "Get into Russian twists position"
        self.exercise_ready = False
        self.stable_frames = 0
        self.stable_threshold = 10   # Need stable position before starting
        
        # Position tracking for proper Russian twists
        self.last_twist_side = "center"
        self.twist_direction = "center"
        self.rep_cooldown = 0
        self.rep_cooldown_max = 15  # frames to wait between reps
        self.position_lost_frames = 0
        self.max_position_lost_frames = 30  # Reset if position lost too long
        
        # Angle tracking for display
        self.current_torso_thigh_angle = 0
        self.current_ground_clearance = 0
        self.current_elbow_knee_distance = {'left': 0, 'right': 0}
        
        # Thresholds for proper Russian twists form
        self.torso_thigh_angle_min = 30   # Minimum V-shape angle
        self.torso_thigh_angle_max = 90   # Maximum V-shape angle
        self.ground_clearance_min = 0.08  # Feet must be this high off ground
        self.elbow_knee_distance_threshold = 0.15  # Max distance for elbow-knee touch
        self.seated_hip_knee_angle_min = 60   # More strict knee bend
        self.seated_hip_knee_angle_max = 120  # More strict knee bend
        
    def reset_exercise(self):
        """Reset all levels for new exercise session"""
        for level in self.levels.values():
            level['passed'] = False
        self.current_level = 1
        self.deepest_level_reached = 1
        self.exercise_ready = False
        self.stable_frames = 0
        self.position_lost_frames = 0
        
    def check_seated_position(self, landmarks):
        """Level 1: Check if person is in seated position with proper knee bend"""
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        left_angle = self.angle_between_points(
            (left_hip.x, left_hip.y), (left_knee.x, left_knee.y), (left_ankle.x, left_ankle.y)
        )
        right_angle = self.angle_between_points(
            (right_hip.x, right_hip.y), (right_knee.x, right_knee.y), (right_ankle.x, right_ankle.y)
        )
        
        return (self.seated_hip_knee_angle_min < left_angle < self.seated_hip_knee_angle_max and
                self.seated_hip_knee_angle_min < right_angle < self.seated_hip_knee_angle_max)
    
    def check_legs_elevated(self, landmarks):
        """Level 2: Check if legs are properly elevated OFF THE GROUND"""
        # Check ground clearance for both feet
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate how high feet are relative to hips (ground reference)
        left_clearance = left_hip.y - left_ankle.y
        right_clearance = right_hip.y - right_ankle.y
        
        # Store for display
        self.current_ground_clearance = round(min(left_clearance, right_clearance), 3)
        
        # Both feet must be significantly above ground level
        return (left_clearance > self.ground_clearance_min and 
                right_clearance > self.ground_clearance_min)
    
    def calculate_torso_thigh_angle(self, landmarks):
        """Calculate the V-shape angle between torso and thighs"""
        # Get torso line (shoulder to hip midpoint)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        mid_shoulder = ((left_shoulder.x + right_shoulder.x) / 2, 
                       (left_shoulder.y + right_shoulder.y) / 2)
        mid_hip = ((left_hip.x + right_hip.x) / 2, 
                  (left_hip.y + right_hip.y) / 2)
        
        # Get thigh line (hip to knee midpoint)
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        mid_knee = ((left_knee.x + right_knee.x) / 2,
                   (left_knee.y + right_knee.y) / 2)
        
        # Calculate angle between torso and thigh lines
        # Vector from hip to shoulder (torso)
        torso_vector = (mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1])
        # Vector from hip to knee (thigh)  
        thigh_vector = (mid_knee[0] - mid_hip[0], mid_knee[1] - mid_hip[1])
        
        # Calculate angle between vectors
        dot_product = (torso_vector[0] * thigh_vector[0] + 
                      torso_vector[1] * thigh_vector[1])
        
        torso_mag = np.sqrt(torso_vector[0]**2 + torso_vector[1]**2)
        thigh_mag = np.sqrt(thigh_vector[0]**2 + thigh_vector[1]**2)
        
        cos_angle = dot_product / (torso_mag * thigh_mag + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        # Store for display
        self.current_torso_thigh_angle = round(angle, 1)
        
        return angle
    
    def check_torso_lean(self, landmarks):
        """Level 3: Check if torso creates proper V-shape with thighs"""
        angle = self.calculate_torso_thigh_angle(landmarks)
        return self.torso_thigh_angle_min < angle < self.torso_thigh_angle_max
    
    def check_stable_position(self, landmarks):
        """Level 4: Check if position is stable and ready for twisting"""
        # Must maintain all previous levels
        return (self.levels[1]['passed'] and 
                self.levels[2]['passed'] and 
                self.levels[3]['passed'] and
                self.check_seated_position(landmarks) and  # Continuous check
                self.check_legs_elevated(landmarks) and    # Continuous check
                self.check_torso_lean(landmarks))          # Continuous check
    
    def calculate_elbow_knee_distances(self, landmarks):
        """Calculate distances between elbows and opposite knees"""
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        # Distance from left elbow to right knee (left twist)
        left_twist_distance = np.sqrt(
            (left_elbow.x - right_knee.x)**2 + 
            (left_elbow.y - right_knee.y)**2
        )
        
        # Distance from right elbow to left knee (right twist)
        right_twist_distance = np.sqrt(
            (right_elbow.x - left_knee.x)**2 + 
            (right_elbow.y - left_knee.y)**2
        )
        
        # Store for display
        self.current_elbow_knee_distance['left'] = round(left_twist_distance, 3)
        self.current_elbow_knee_distance['right'] = round(right_twist_distance, 3)
        
        return left_twist_distance, right_twist_distance
    
    def detect_proper_twist(self, landmarks):
        """Detect proper Russian twist with elbow-to-knee movement"""
        left_dist, right_dist = self.calculate_elbow_knee_distances(landmarks)
        
        # Determine current twist state
        current_twist = "center"
        
        if left_dist < self.elbow_knee_distance_threshold:
            current_twist = "left"  # Left elbow touching right knee
        elif right_dist < self.elbow_knee_distance_threshold:
            current_twist = "right"  # Right elbow touching left knee
        
        return current_twist
    
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
    
    def draw_angle_visualization(self, frame, landmarks):
        """Draw comprehensive visualization of Russian twist angles"""
        h, w, _ = frame.shape
        
        # Get landmark positions
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Calculate midpoints
        mid_shoulder = (int((left_shoulder.x + right_shoulder.x) / 2 * w), 
                       int((left_shoulder.y + right_shoulder.y) / 2 * h))
        mid_hip = (int((left_hip.x + right_hip.x) / 2 * w), 
                  int((left_hip.y + right_hip.y) / 2 * h))
        mid_knee = (int((left_knee.x + right_knee.x) / 2 * w),
                   int((left_knee.y + right_knee.y) / 2 * h))
        
        # Convert to pixel coordinates
        left_elbow_px = (int(left_elbow.x * w), int(left_elbow.y * h))
        right_elbow_px = (int(right_elbow.x * w), int(right_elbow.y * h))
        left_knee_px = (int(left_knee.x * w), int(left_knee.y * h))
        right_knee_px = (int(right_knee.x * w), int(right_knee.y * h))
        left_ankle_px = (int(left_ankle.x * w), int(left_ankle.y * h))
        right_ankle_px = (int(right_ankle.x * w), int(right_ankle.y * h))
        
        # Draw torso line (shoulder to hip)
        cv2.line(frame, mid_shoulder, mid_hip, (255, 0, 255), 3)
        cv2.putText(frame, "TORSO", (mid_shoulder[0]-30, mid_shoulder[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Draw thigh line (hip to knee)
        cv2.line(frame, mid_hip, mid_knee, (0, 255, 255), 3)
        cv2.putText(frame, "THIGH", (mid_knee[0]-30, mid_knee[1]+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw elbow-to-knee lines for twist detection
        cv2.line(frame, left_elbow_px, right_knee_px, (0, 255, 0), 2)  # Left twist
        cv2.line(frame, right_elbow_px, left_knee_px, (255, 0, 0), 2)  # Right twist
        
        # Highlight feet to show ground clearance
        cv2.circle(frame, left_ankle_px, 8, (255, 255, 0), -1)
        cv2.circle(frame, right_ankle_px, 8, (255, 255, 0), -1)
        
        # Draw angle arc at hip joint
        if self.current_torso_thigh_angle > 10:
            cv2.ellipse(frame, mid_hip, (60, 60), 0, 0, int(self.current_torso_thigh_angle), 
                       (255, 255, 255), 2)
        
        # Add angle text
        cv2.putText(frame, f"V-Angle: {self.current_torso_thigh_angle}°", 
                   (mid_hip[0] + 70, mid_hip[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def update_levels(self, landmarks):
        """Update the progression through levels with strict position maintenance"""
        rep_completed = False
        
        # Always calculate angles for display
        self.calculate_torso_thigh_angle(landmarks)
        self.calculate_elbow_knee_distances(landmarks)
        
        # Check if basic position is maintained (strict checking)
        position_maintained = (self.check_seated_position(landmarks) and 
                             self.check_legs_elevated(landmarks) and 
                             self.check_torso_lean(landmarks))
        
        if not position_maintained and self.exercise_ready:
            self.position_lost_frames += 1
            if self.position_lost_frames > self.max_position_lost_frames:
                # Reset exercise if position lost for too long
                self.reset_exercise()
                self.feedback_message = "Position lost! Get back into Russian twists position"
                return False
        else:
            self.position_lost_frames = 0
        
        # Level 1: Seated Position
        if self.current_level >= 1 and not self.levels[1]['passed']:
            if self.check_seated_position(landmarks):
                self.levels[1]['passed'] = True
                self.deepest_level_reached = max(self.deepest_level_reached, 1)
        
        # Level 2: Legs Elevated OFF THE GROUND
        if self.levels[1]['passed'] and not self.levels[2]['passed']:
            if self.check_legs_elevated(landmarks):
                self.levels[2]['passed'] = True
                self.deepest_level_reached = max(self.deepest_level_reached, 2)
        
        # Level 3: Proper V-shape angle between torso and thighs
        if self.levels[2]['passed'] and not self.levels[3]['passed']:
            if self.check_torso_lean(landmarks):
                self.levels[3]['passed'] = True
                self.deepest_level_reached = max(self.deepest_level_reached, 3)
        
        # Level 4: Stable position maintenance
        if self.levels[3]['passed'] and not self.levels[4]['passed']:
            if self.check_stable_position(landmarks):
                self.stable_frames += 1
                if self.stable_frames >= self.stable_threshold:
                    self.levels[4]['passed'] = True
                    self.exercise_ready = True
                    self.deepest_level_reached = max(self.deepest_level_reached, 4)
            else:
                self.stable_frames = 0
        
        # Level 5: Proper Russian twist motion (elbow to knee)
        if self.exercise_ready and self.levels[4]['passed'] and position_maintained:
            self.levels[5]['passed'] = True
            self.deepest_level_reached = max(self.deepest_level_reached, 5)
            
            # Detect proper twist motion
            current_twist = self.detect_proper_twist(landmarks)
            
            # Decrease cooldown
            if self.rep_cooldown > 0:
                self.rep_cooldown -= 1
            
            # Count reps only for proper elbow-to-knee touches
            if self.rep_cooldown == 0:
                if ((self.last_twist_side == "left" and current_twist == "right") or
                    (self.last_twist_side == "right" and current_twist == "left")):
                    rep_completed = True
                    self.rep_cooldown = self.rep_cooldown_max
                
                # Update last twist side
                if current_twist != "center":
                    self.last_twist_side = current_twist
        
        # Update feedback message
        self.update_feedback_message()
        return rep_completed
    
    def update_feedback_message(self):
        """Update feedback message based on current progress"""
        if self.position_lost_frames > 0:
            self.feedback_message = f"⚠️ Maintain position! ({self.position_lost_frames}/{self.max_position_lost_frames})"
        elif not self.levels[1]['passed']:
            self.feedback_message = "⚠️ " + self.levels[1]['message']
        elif not self.levels[2]['passed']:
            self.feedback_message = f"⚠️ {self.levels[2]['message']} (clearance: {self.current_ground_clearance})"
        elif not self.levels[3]['passed']:
            self.feedback_message = f"⚠️ {self.levels[3]['message']} (angle: {self.current_torso_thigh_angle}°)"
        elif not self.levels[4]['passed']:
            remaining = max(0, self.stable_threshold - self.stable_frames)
            if remaining > 0:
                self.feedback_message = f"⚠️ Hold position steady ({remaining} more)"
            else:
                self.feedback_message = "✅ Position ready! Start twisting"
        elif self.levels[5]['passed']:
            left_dist = self.current_elbow_knee_distance['left']
            right_dist = self.current_elbow_knee_distance['right']
            self.feedback_message = f"✅ Touch elbow to knee! L:{left_dist:.2f} R:{right_dist:.2f}"
        else:
            self.feedback_message = "⚠️ " + self.levels[5]['message']
    
    def get_debug_info(self):
        """Get debugging information for display"""
        debug_info = []
        debug_info.append(f"Exercise Ready: {self.exercise_ready}")
        debug_info.append(f"Position Lost Frames: {self.position_lost_frames}")
        debug_info.append(f"Stable Frames: {self.stable_frames}/{self.stable_threshold}")
        debug_info.append(f"Last Twist: {self.last_twist_side}")
        debug_info.append(f"Rep Cooldown: {self.rep_cooldown}")
        debug_info.append(f"Torso-Thigh Angle: {self.current_torso_thigh_angle}°")
        debug_info.append(f"Ground Clearance: {self.current_ground_clearance}")
        debug_info.append(f"Elbow-Knee L: {self.current_elbow_knee_distance['left']:.3f}")
        debug_info.append(f"Elbow-Knee R: {self.current_elbow_knee_distance['right']:.3f}")
        
        for level_num, level_data in self.levels.items():
            status = "✅" if level_data['passed'] else "❌"
            debug_info.append(f"L{level_num} {level_data['name']}: {status}")
            
        return debug_info
    
    def get_feedback(self):
        """Get current feedback message"""
        return self.feedback_message

# Create feedback tracker instance
feedback_tracker = RussianTwistsFeedbackTracker()

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
        
        label = "Russian Twists Position"
        
        # Update multi-level feedback system
        rep_completed = feedback_tracker.update_levels(landmarks)
        
        if rep_completed:
            rep_count += 1

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw comprehensive angle visualization
        feedback_tracker.draw_angle_visualization(frame, landmarks)

    else:
        feedback_tracker.feedback_message = "Position yourself in view for Russian twists"
        label = "No pose detected"

    # Display main info
    cv2.putText(frame, f'Exercise: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
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
        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25

    # Display debug information
    debug_info = feedback_tracker.get_debug_info()
    debug_y_offset = 200
    cv2.putText(frame, "=== DEBUG INFO ===", (10, debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
    debug_y_offset += 20
    
    for debug_line in debug_info:
        cv2.putText(frame, debug_line, (10, debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        debug_y_offset += 18

    # Show frame
    cv2.imshow("Russian Twists Detector", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()