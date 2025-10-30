Golf ball video tracking with stroke detection:
 - Loads a golf video file
 - Runs YOLOv8 detector per frame focusing on sports balls and people
 - Detects golf swings based on ball movement patterns
 - Tracks each stroke and exports data to JSON
 - Annotates frames with golf ball track ids and stroke information

Requirements (pip):
  pip install ultralytics trackers supervision opencv-python

Usage:
  python gtracker.py input.mp4 
  output_tracked_golf.mp4