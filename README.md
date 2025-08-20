# Traffic_Flow_Analysis
Traffic Flow Analysis using YOLOv8 &amp; DeepSORT - Automated vehicle detection, tracking, and per-lane counting from video streams.
This project implements a computer vision pipeline for traffic flow analysis. It:
Downloads traffic videos directly from YouTube
Uses YOLOv8 (Ultralytics) for vehicle detection (cars, buses, trucks, motorcycles)
Tracks vehicles using DeepSORT to maintain unique IDs across frames
Defines lane polygons and counts vehicles per lane
Generates outputs including:
🎥 Annotated video with bounding boxes & counts
📊 events.csv with detailed logs (vehicle ID, lane, timestamp)
📄 summary.json with total counts per lane
