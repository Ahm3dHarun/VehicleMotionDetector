import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial import distance

class BasicKalmanFilter:
    def __init__(self, x, y):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0], 
                              [0, 1, 0, 1], 
                              [0, 0, 1, 0], 
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], 
                              [0, 1, 0, 0]])
        self.kf.P *= 1000  # High uncertainty in the initial state
        self.kf.R *= 10    # Measurement noise
        self.kf.Q = np.eye(4)  # Process noise
        self.kf.x = np.array([x, y, 0, 0])  # Initial state
        self.history = [(x, y)]

    def predict(self):
        self.kf.predict()
        return self.kf.x[:2]

    def update(self, x, y):
        self.kf.update([x, y])
        self.history.append((x, y))

    def get_state(self):
        return self.kf.x

    def get_history(self):
        return self.history

class MotionDetection:
    def __init__(self, frame_hysteresis=5, motion_threshold=50, distance_threshold=50, skip_frames=1, max_objects=10):
        self.frame_hysteresis = frame_hysteresis
        self.motion_threshold = motion_threshold
        self.distance_threshold = distance_threshold
        self.skip_frames = skip_frames
        self.max_objects = max_objects

        self.trackers = []
        self.frame_count = 0

    def initialize_tracker(self, initial_frame):
        gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.fgmask = self.fgbg.apply(gray)

    def update(self, frame):
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = self.fgbg.apply(gray)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_detections = []

        for contour in contours:
            if cv2.contourArea(contour) > self.motion_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
                new_detections.append((cx, cy))

        # Update existing trackers
        for tracker in self.trackers:
            state = tracker.predict()
            tracker.updated = False
            for det in new_detections:
                if distance.euclidean(state, det) < self.distance_threshold:
                    tracker.update(det[0], det[1])
                    tracker.updated = True
                    new_detections.remove(det)
                    break

        # Remove inactive trackers
        self.trackers = [tracker for tracker in self.trackers if tracker.updated or self.frame_count - tracker.last_update < self.frame_hysteresis]

        # Add new trackers
        for det in new_detections:
            if len(self.trackers) < self.max_objects:
                new_tracker = BasicKalmanFilter(det[0], det[1])
                self.trackers.append(new_tracker)

    def get_state(self):
        return [(int(tracker.get_state()[0]), int(tracker.get_state()[1])) for tracker in self.trackers]

    def get_histories(self):
        return [tracker.get_history() for tracker in self.trackers]

def main():
    cap = cv2.VideoCapture('video.mp4')  # Replace 'video.mp4' with your video file

    ret, frame = cap.read()
    motion_detector = MotionDetection()
    if ret:
        motion_detector.initialize_tracker(frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        motion_detector.update(frame)
        states = motion_detector.get_state()

        for (x, y) in states:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow('Vehicle Detection', frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
