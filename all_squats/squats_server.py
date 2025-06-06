from flask import Flask, jsonify, request
import threading
import time
import cv2
import mediapipe as mp
import numpy as np

# Import your SquatFeedbackTracker from squats_main.py
from squats_main import SquatFeedbackTracker

app = Flask(__name__)

# Shared state for detection results
detection_result = {"reps": 0, "feedback": "Not started", "active": False}
detection_thread = None
stop_flag = False


def run_squat_detection():
    global detection_result, stop_flag
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    feedback_tracker = SquatFeedbackTracker()
    rep_count = 0
    cap = cv2.VideoCapture(0)
    detection_result["active"] = True
    while cap.isOpened() and not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            rep_completed = feedback_tracker.update(landmarks)
            if rep_completed:
                rep_count += 1
            feedback = feedback_tracker.get_feedback()
        else:
            feedback = "Position yourself in view for squats"
        detection_result["reps"] = rep_count
        detection_result["feedback"] = feedback
        time.sleep(0.05)  # Reduce CPU usage
    cap.release()
    detection_result["active"] = False
    stop_flag = False


@app.route('/start-squat-detection', methods=['POST'])
def start_detection():
    global detection_thread, stop_flag
    if detection_thread is None or not detection_thread.is_alive():
        stop_flag = False
        detection_thread = threading.Thread(target=run_squat_detection)
        detection_thread.start()
        return jsonify({"status": "started"})
    else:
        return jsonify({"status": "already running"})


@app.route('/stop-squat-detection', methods=['POST'])
def stop_detection():
    global stop_flag
    stop_flag = True
    return jsonify({"status": "stopping"})


@app.route('/get-squat-status', methods=['GET'])
def get_status():
    return jsonify(detection_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
