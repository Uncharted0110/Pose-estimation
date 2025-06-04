from flask import Flask, request, jsonify, session
from pymongo import MongoClient
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import base64
import io
from PIL import Image
from datetime import datetime


app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# Load model, scaler, and setup (same as your original code)
model = joblib.load("pushup_detector.pkl")
scaler = joblib.load("scaler.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mongo_client = MongoClient('mongodb+srv://exercia:Fpd20jbSjavHHM5H@cluster0.qpv0m.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = mongo_client['test']
users_collection = db['users']

# Global session state (in production, use Redis or database)
sessions = {}

def add_workout_to_user(user_id, reps, workout_name, time_taken):
    workout_entry = {
        "reps": reps,
        "workout_name": workout_name,
        "time_taken": time_taken,
        "date": datetime.utcnow()
    }
    users_collection.update_one(
        {"_id": user_id},
        {"$push": {"workouts": workout_entry}}
    )

class MultiLevelFeedbackTracker:
    def __init__(self):
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
        self.body_straight_threshold = 3  # Reduced from 10 to 3
        
        # Adjusted thresholds for better detection
        self.straight_body_angle_threshold = 140  # More lenient from 160
        self.descent_threshold = 0.02  # Reduced from 0.05
        self.depth_threshold = 0.15   # Much more realistic from 0.6
        self.ascent_threshold = 0.05  # More realistic from 0.5
        
        # Add position tracking
        self.initial_nose_y = None
        self.min_nose_y = None  # Track lowest point
        self.max_descent = 0
        
    def reset_rep(self):
        print("Resetting rep...")
        for level in self.levels.values():
            level['passed'] = False
        self.current_level = 1
        self.deepest_level_reached = 1
        self.rep_in_progress = False
        self.descent_started_y = None
        self.body_straight_frames = 0
        self.initial_nose_y = None
        self.min_nose_y = None
        self.max_descent = 0
        
    def check_body_alignment(self, landmarks):
        try:
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
            print(f"Back angle: {back_angle}")
            
            if back_angle >= self.straight_body_angle_threshold:
                self.body_straight_frames += 1
                print(f"Body straight frames: {self.body_straight_frames}")
                if self.body_straight_frames >= self.body_straight_threshold:
                    return True
            else:
                self.body_straight_frames = max(0, self.body_straight_frames - 1)  # Gradual decrease
                
            return False
        except Exception as e:
            print(f"Error in body alignment check: {e}")
            return False
    
    def check_descent_started(self, nose_y):
        if self.initial_nose_y is None:
            self.initial_nose_y = nose_y
            print(f"Initial nose Y: {self.initial_nose_y}")
            return False
            
        descent_amount = nose_y - self.initial_nose_y
        print(f"Descent amount: {descent_amount}, threshold: {self.descent_threshold}")
        
        if descent_amount >= self.descent_threshold:
            return True
        return False
    
    def check_depth_reached(self, nose_y):
        if self.initial_nose_y is None:
            return False
            
        descent_amount = nose_y - self.initial_nose_y
        self.max_descent = max(self.max_descent, descent_amount)
        print(f"Current descent: {descent_amount}, max: {self.max_descent}, threshold: {self.depth_threshold}")
        
        return descent_amount >= self.depth_threshold
    
    def check_ascent_complete(self, nose_y):
        if self.initial_nose_y is None or self.max_descent == 0:
            return False
            
        current_descent = nose_y - self.initial_nose_y
        ascent_progress = (self.max_descent - current_descent) / self.max_descent
        print(f"Ascent progress: {ascent_progress}")
        
        # Complete when back to 80% of original position
        return ascent_progress >= 0.8
    
    def angle_between_points(self, p1, p2, p3):
        try:
            a = np.array(p1)
            b = np.array(p2)
            c = np.array(p3)

            ba = a - b
            bc = c - b

            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 0
    
    def update_levels(self, landmarks, nose_y, is_pushup_detected):
        print(f"Update levels - Pushup detected: {is_pushup_detected}, Current level: {self.current_level}")
        
        if not is_pushup_detected:
            if self.rep_in_progress:
                print("Lost pushup detection, resetting...")
                self.reset_rep()
            self.feedback_message = "Position yourself for pushup"
            return False
        
        # Level 1: Body Alignment
        if not self.rep_in_progress:
            if self.check_body_alignment(landmarks):
                print("Body alignment achieved!")
                self.rep_in_progress = True
                self.levels[1]['passed'] = True
                self.current_level = 2
                self.deepest_level_reached = max(self.deepest_level_reached, 1)
                self.initial_nose_y = nose_y  # Set baseline
        
        # Level 2: Descent Started
        if self.rep_in_progress and self.current_level == 2 and not self.levels[2]['passed']:
            if self.check_descent_started(nose_y):
                print("Descent started!")
                self.levels[2]['passed'] = True
                self.current_level = 3
                self.deepest_level_reached = max(self.deepest_level_reached, 2)
        
        # Level 3: Depth Reached
        if self.current_level == 3 and not self.levels[3]['passed']:
            if self.check_depth_reached(nose_y):
                print("Depth reached!")
                self.levels[3]['passed'] = True
                self.current_level = 4
                self.deepest_level_reached = max(self.deepest_level_reached, 3)
        
        # Level 4: Ascent Complete
        if self.current_level == 4 and not self.levels[4]['passed']:
            if self.check_ascent_complete(nose_y):
                print("Ascent complete! Rep finished!")
                self.levels[4]['passed'] = True
                self.deepest_level_reached = max(self.deepest_level_reached, 4)
                self.feedback_message = "✅ Excellent Form! Rep Completed!"
                return True
        
        self.update_feedback_message()
        self.last_nose_y = nose_y
        return False
    
    def update_feedback_message(self):
        if self.deepest_level_reached == 1 and self.levels[1]['passed']:
            self.feedback_message = "✅ Body aligned! Start lowering down"
        elif self.deepest_level_reached == 2 and self.levels[2]['passed']:
            self.feedback_message = "✅ Descent started! Go lower"
        elif self.deepest_level_reached == 3 and self.levels[3]['passed']:
            self.feedback_message = "✅ Good depth! Push back up"
        else:
            if not self.levels[1]['passed']:
                self.feedback_message = "⚠️ " + self.levels[1]['message']
            elif not self.levels[2]['passed']:
                self.feedback_message = "⚠️ " + self.levels[2]['message']
            elif not self.levels[3]['passed']:
                self.feedback_message = "⚠️ " + self.levels[3]['message']
            elif not self.levels[4]['passed']:
                self.feedback_message = "⚠️ " + self.levels[4]['message']
    
    def get_feedback(self):
        return self.feedback_message

@app.route('/start_session', methods=['POST'])
def start_session():
    session_id = request.json.get('session_id', 'default')
    sessions[session_id] = {
        'tracker': MultiLevelFeedbackTracker(),
        'prediction_history': deque(maxlen=20),  # Reduced from 50
        'rep_count': 0
    }
    print(f"Started session: {session_id}")
    return jsonify({'status': 'success', 'session_id': session_id})

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        session_id = data.get('session_id', 'default')
        image_data = data.get('image')

        if session_id not in sessions:
            print(f"Session {session_id} not found")
            return jsonify({'error': 'Session not found'}), 400

        # Decode base64 image with better error handling
        try:
            # Handle data URL format
            if ',' in image_data:
                base64_data = image_data.split(',')[1]
            else:
                base64_data = image_data

            # Add padding if needed
            missing_padding = len(base64_data) % 4
            if missing_padding:
                base64_data += '=' * (4 - missing_padding)

            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        except Exception as decode_error:
            print(f"Image decode error: {decode_error}")
            return jsonify({'error': 'Failed to decode image data'}), 400

        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        response = {
            'pose_detected': False,
            'feedback': 'Position yourself in view',
            'rep_count': sessions[session_id]['rep_count'],
            'landmarks': []
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            pose_data = []

            # Extract landmark data
            for lm in landmarks:
                pose_data.extend([lm.x, lm.y, lm.z, lm.visibility])

            # Predict
            x = scaler.transform([pose_data])
            pred = model.predict(x)[0]
            sessions[session_id]['prediction_history'].append(pred)

            print(f"Prediction: {pred}, History length: {len(sessions[session_id]['prediction_history'])}")

            # Voting logic
            if len(sessions[session_id]['prediction_history']) >= 5:
                recent_predictions = list(sessions[session_id]['prediction_history'])[-10:]
                vote = sum(recent_predictions)
                is_pushup = vote > len(recent_predictions) * 0.4

                print(f"Vote: {vote}/{len(recent_predictions)}, Is pushup: {is_pushup}")

                nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                rep_completed = sessions[session_id]['tracker'].update_levels(landmarks, nose.y, is_pushup)

                if rep_completed:
                    sessions[session_id]['rep_count'] += 1
                    print(f"Rep completed! Total: {sessions[session_id]['rep_count']}")
                    sessions[session_id]['tracker'].reset_rep()

                response.update({
                    'pose_detected': bool(is_pushup),
                    'feedback': sessions[session_id]['tracker'].get_feedback(),
                    'rep_count': int(sessions[session_id]['rep_count']),
                    'landmarks': [[float(lm.x), float(lm.y)] for lm in landmarks]
                })
            else:
                response['feedback'] = 'Analyzing pose...'

        return jsonify(response)

    except Exception as e:
        print(f"Error in analyze_frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    session_id = request.json.get('session_id', 'default')
    if session_id in sessions:
        sessions[session_id]['rep_count'] = 0
        sessions[session_id]['tracker'].reset_rep()
        sessions[session_id]['prediction_history'].clear()
        print(f"Reset session: {session_id}")
    return jsonify({'status': 'success'})

@app.route('/debug_session', methods=['GET'])
def debug_session():
    """Debug endpoint to check session states"""
    debug_info = {}
    for session_id, session in sessions.items():
        tracker = session['tracker']
        debug_info[session_id] = {
            'rep_count': session['rep_count'],
            'current_level': tracker.current_level,
            'rep_in_progress': tracker.rep_in_progress,
            'levels_passed': {k: v['passed'] for k, v in tracker.levels.items()},
            'feedback': tracker.get_feedback(),
            'prediction_history_length': len(session['prediction_history'])
        }
    return jsonify(debug_info)

@app.route('/end_session', methods=['POST'])
def end_session():
    data = request.json
    session_id = data.get('session_id', 'default')
    user_id = data.get('user_id')
    workout_name = data.get('workout_name', 'pushup')
    time_taken = data.get('time_taken', 0)

    print(f"Ending session: {session_id} for user: {user_id}")
    print(f"Workout name: {workout_name}, Time taken: {time_taken}, Reps: {sessions[session_id]['rep_count']}")
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User ID is required'}), 400
    if not workout_name:
        return jsonify({'status': 'error', 'message': 'Workout name is required'}), 400
    if not time_taken:
        return jsonify({'status': 'error', 'message': 'Time taken is required'}), 400
    
    if session_id in sessions:
        rep_count = sessions[session_id]['rep_count']
        # Add workout to user's history in MongoDB
        workout_entry = {
            "reps": rep_count,
            "workout_name": workout_name,
            "time_taken": time_taken,
            "date": datetime.utcnow()
        }
        users_collection.update_one(
            {"email": user_id},
            {"$push": {"workouts": workout_entry}}
        )
        # Optionally, remove session from memory
        del sessions[session_id]
        return jsonify({'status': 'success', 'reps': rep_count})
    else:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)