import sys
import json
import math
from datetime import datetime
from ultralytics import YOLO
import cv2
import numpy as np

from trackers.core.sort.tracker import SORTTracker
import supervision as sv

# -------- GOLF TRACKING CONFIG ----------
MODEL_PATH = "yolov8n.pt"   # Use YOLOv8 trained model
CONF_THRESH = 0.25          # Lower threshold for small golf balls
IOU_THRESH = 0.2            # Lower IOU for small object tracking
MIN_BALL_SIZE = 5           # Minimum pixel size for golf ball detection
MAX_BALL_SIZE = 100         # Maximum pixel size for golf ball detection
SPORTS_BALL_CLASS_ID = 32   # COCO class ID for sports ball (includes golf balls)
PERSON_CLASS_ID = 0         # COCO class ID for person

# Swing detection parameters
MIN_SPEED_FOR_STROKE = 50   # Minimum pixel speed to consider a stroke
STROKE_COOLDOWN_FRAMES = 30 # Frames to wait before detecting next stroke
MAX_STROKE_DISTANCE = 300   # Maximum distance ball can travel in one stroke detection
# -----------------------------------------

class StrokeTracker:
    def __init__(self):
        self.strokes = []
        self.current_stroke = None
        self.last_stroke_frame = 0
        self.ball_positions = {}
        
    def update_ball_position(self, track_id, x, y, frame_idx):
        if track_id not in self.ball_positions:
            self.ball_positions[track_id] = []
        
        self.ball_positions[track_id].append((x, y, frame_idx))
        
        if len(self.ball_positions[track_id]) > 10:
            self.ball_positions[track_id] = self.ball_positions[track_id][-10:]
    
    def detect_stroke(self, track_id, frame_idx, fps):
        if track_id not in self.ball_positions:
            return False
            
        positions = self.ball_positions[track_id]
        if len(positions) < 3:
            return False
            
        if frame_idx - self.last_stroke_frame < STROKE_COOLDOWN_FRAMES:
            return False
        
        recent_positions = positions[-3:]
        if len(recent_positions) < 3:
            return False
            
        speeds = []
        for i in range(1, len(recent_positions)):
            x1, y1, f1 = recent_positions[i-1]
            x2, y2, f2 = recent_positions[i]
            
            distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            time_diff = (f2 - f1) / fps if fps > 0 else 1
            speed = distance / time_diff if time_diff > 0 else 0
            speeds.append(speed)
        
        if len(speeds) >= 2:
            speed_increase = speeds[-1] - speeds[-2] if len(speeds) > 1 else speeds[-1]
            if speed_increase > MIN_SPEED_FOR_STROKE:
                return True
                
        return False
    
    def start_stroke(self, track_id, frame_idx, ball_pos, timestamp):
        self.current_stroke = {
            "stroke_id": len(self.strokes) + 1,
            "ball_track_id": track_id,
            "start_frame": frame_idx,
            "start_position": ball_pos,
            "start_timestamp": timestamp,
            "ball_path": [ball_pos],
            "end_frame": None,
            "end_position": None,
            "distance": 0,
            "max_speed": 0,
            "duration_seconds": 0
        }
        self.last_stroke_frame = frame_idx
        
    def update_stroke(self, track_id, frame_idx, ball_pos, fps):
        if self.current_stroke and self.current_stroke["ball_track_id"] == track_id:
            self.current_stroke["ball_path"].append(ball_pos)
            self.current_stroke["end_frame"] = frame_idx
            self.current_stroke["end_position"] = ball_pos
            
            start_pos = self.current_stroke["start_position"]
            distance = math.sqrt((ball_pos[0] - start_pos[0])**2 + (ball_pos[1] - start_pos[1])**2)
            self.current_stroke["distance"] = distance
            
            frame_diff = frame_idx - self.current_stroke["start_frame"]
            self.current_stroke["duration_seconds"] = frame_diff / fps if fps > 0 else 0
            
            if len(self.current_stroke["ball_path"]) >= 2:
                path = self.current_stroke["ball_path"]
                max_speed = 0
                for i in range(1, len(path)):
                    dist = math.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
                    speed = dist * fps
                    max_speed = max(max_speed, speed)
                self.current_stroke["max_speed"] = max_speed
    
    def end_stroke(self):
        if self.current_stroke:
            self.current_stroke["end_timestamp"] = datetime.now().isoformat()
            self.strokes.append(self.current_stroke.copy())
            self.current_stroke = None
    
    def should_end_stroke(self, track_id, frame_idx):
        if not self.current_stroke or self.current_stroke["ball_track_id"] != track_id:
            return False
            
        frame_duration = frame_idx - self.current_stroke["start_frame"]
        distance = self.current_stroke["distance"]
        
        return frame_duration > 120 or distance > MAX_STROKE_DISTANCE

def is_likely_golf_ball(box, class_id, confidence):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    if class_id != SPORTS_BALL_CLASS_ID:
        return False
    
    if width < MIN_BALL_SIZE or height < MIN_BALL_SIZE:
        return False
    if width > MAX_BALL_SIZE or height > MAX_BALL_SIZE:
        return False
    
    aspect_ratio = width / height if height > 0 else 0
    if aspect_ratio < 0.7 or aspect_ratio > 1.3:
        return False
    
    return True

def boxes_from_yolo_result(results, conf_thresh=CONF_THRESH):
    detections = []
    for box, score, cls in zip(results.boxes.xyxy.cpu().numpy(),
                               results.boxes.conf.cpu().numpy(),
                               results.boxes.cls.cpu().numpy()):
        if score < conf_thresh:
            continue
        
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        
        if class_id == PERSON_CLASS_ID or is_likely_golf_ball([x1, y1, x2, y2], class_id, score):
            detections.append([x1, y1, x2, y2, float(score), class_id])
    
    return detections

def save_strokes_to_json(strokes, output_path, video_info):
    json_path = output_path.replace('.mp4', '_strokes.json')
    
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    serializable_strokes = convert_numpy_types(strokes)
    serializable_video_info = convert_numpy_types(video_info)
    
    stroke_data = {
        "video_info": serializable_video_info,
        "analysis_timestamp": datetime.now().isoformat(),
        "total_strokes": len(serializable_strokes),
        "strokes": serializable_strokes
    }
    
    with open(json_path, 'w') as f:
        json.dump(stroke_data, f, indent=2)
    
    print(f"Stroke data saved to: {json_path}")
    return json_path

def main(in_path, out_path):
    model = YOLO(MODEL_PATH)

    tracker = SORTTracker()
    
    stroke_tracker = StrokeTracker()

    box_annotator = sv.BoxAnnotator(
        color=sv.Color.GREEN,
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.Color.GREEN,
        text_color=sv.Color.WHITE
    )

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open golf video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    total_golf_balls_detected = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(rgb, verbose=False)[0]

        dets = boxes_from_yolo_result(results, conf_thresh=CONF_THRESH)
        
        if len(dets) > 0:
            golf_ball_dets = [d for d in dets if d[5] == SPORTS_BALL_CLASS_ID]
            total_golf_balls_detected += len(golf_ball_dets)

        if len(dets) > 0:
            dets_arr = np.array(dets)
            detections = sv.Detections(
                xyxy=dets_arr[:, :4],
                confidence=dets_arr[:, 4],
                class_id=dets_arr[:, 5].astype(int)
            )
        else:
            detections = sv.Detections.empty()

        tracks = tracker.update(detections)

        for i, track_id in enumerate(tracks.tracker_id if len(tracks) > 0 else []):
            class_id = tracks.class_id[i]
            if class_id == SPORTS_BALL_CLASS_ID:
                bbox = tracks.xyxy[i]
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                stroke_tracker.update_ball_position(track_id, center_x, center_y, frame_idx)
                
                if stroke_tracker.detect_stroke(track_id, frame_idx, fps):
                    if not stroke_tracker.current_stroke:
                        stroke_tracker.start_stroke(track_id, frame_idx, (center_x, center_y), datetime.now().isoformat())
                        print(f"Stroke detected at frame {frame_idx}!")
                
                if stroke_tracker.current_stroke and stroke_tracker.current_stroke["ball_track_id"] == track_id:
                    stroke_tracker.update_stroke(track_id, frame_idx, (center_x, center_y), fps)
                    
                    if stroke_tracker.should_end_stroke(track_id, frame_idx):
                        stroke_tracker.end_stroke()
                        print(f"Stroke ended at frame {frame_idx}")

        labels = []
        if len(tracks) > 0:
            for i, (track_id, class_id) in enumerate(zip(tracks.tracker_id, tracks.class_id)):
                if class_id == SPORTS_BALL_CLASS_ID:
                    label = f"Golf Ball #{track_id}"
                    if stroke_tracker.current_stroke and stroke_tracker.current_stroke["ball_track_id"] == track_id:
                        label += " [STROKE]"
                    labels.append(label)
                elif class_id == PERSON_CLASS_ID:
                    labels.append(f"Player #{track_id}")
                else:
                    labels.append(f"Object #{track_id}")

        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=tracks
        )
        
        if len(tracks) > 0:
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=tracks,
                labels=labels
            )

        stroke_info = f"Strokes: {len(stroke_tracker.strokes)}"
        if stroke_tracker.current_stroke:
            stroke_info += " [IN PROGRESS]"
            
        cv2.putText(annotated_frame, f"Frame: {frame_idx} | Golf Balls: {len([t for t in tracks.class_id if t == SPORTS_BALL_CLASS_ID])} | {stroke_info}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(annotated_frame)

        if frame_idx % 30 == 0:
            print(f"Processing frame {frame_idx}... Strokes detected: {len(stroke_tracker.strokes)}")

    if stroke_tracker.current_stroke:
        stroke_tracker.end_stroke()

    cap.release()
    out.release()
    
    video_info = {
        "input_file": in_path,
        "output_file": out_path,
        "fps": fps,
        "width": w,
        "height": h,
        "total_frames": frame_idx
    }
    
    json_path = save_strokes_to_json(stroke_tracker.strokes, out_path, video_info)
    
    print(f"Golf stroke tracking complete!")
    print(f"Total strokes detected: {len(stroke_tracker.strokes)}")
    print(f"Total golf ball detections: {total_golf_balls_detected}")
    print(f"Saved tracked golf video: {out_path}")
    print(f"Saved stroke data: {json_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python golf_ball_tracker.py input_golf.mp4 output_tracked_golf.mp4")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
