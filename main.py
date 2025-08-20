import os
import cv2
import json
import math
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from yt_dlp import YoutubeDL


# Optional GPU selection via env var, but code auto-detects via torch inside ultralytics

# -------------------------
# Helpers: YouTube download
# -------------------------

def download_youtube(url, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / "%(title)s.%(ext)s")
    
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': output_path
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return out_dir

# -------------------------
# Helpers: Geometry & Lanes
# -------------------------

def point_in_polygon(point, polygon):
    """Return True if point (x,y) lies inside polygon [(x1,y1),...]. Uses cv2.pointPolygonTest."""
    contour = np.array(polygon, dtype=np.int32)
    res = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
    return res >= 0


def make_default_lanes(w: int, h: int):
    """
    Define 3 lane polygons based on relative frame size. Adjust as needed per video.
    Lanes are vertical bands covering bottom 70% of the frame to mitigate sky/overpass noise.
    """
    y_top = int(h * 0.30)
    x1 = int(w * 0.00)
    x2 = int(w * 0.33)
    x3 = int(w * 0.66)
    x4 = int(w * 1.00)
    lane1 = [(x1, y_top), (x2, y_top), (x2, h), (x1, h)]
    lane2 = [(x2, y_top), (x3, y_top), (x3, h), (x2, h)]
    lane3 = [(x3, y_top), (x4, y_top), (x4, h), (x3, h)]
    return [lane1, lane2, lane3]


# -------------------------
# Tracking wrapper (DeepSORT)
# -------------------------

def build_tracker(max_age=30, n_init=3):
    from deep_sort_realtime.deepsort_tracker import DeepSort

    # Metric and appearance settings are left mostly default for simplicity
    tracker = DeepSort(
        max_age=max_age,
        n_init=n_init,
        max_iou_distance=0.7,
        max_cosine_distance=0.3,
        nms_max_overlap=1.0,
        embedder="mobilenet",
        half=True,
        bgr=True,
    )
    return tracker


# -------------------------
# Detection wrapper (YOLO)
# -------------------------

def load_detector(model_path: str = "yolov8n.pt"):
    from ultralytics import YOLO

    model = YOLO(model_path)
    return model


VEHICLE_CLS = {
    2: "car",  # COCO class indices for YOLOv8
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# -------------------------
# Main processing loop
# -------------------------

def process_video(
    video_path: Path,
    output_dir: Path,
    model_path: str = "yolov8n.pt",
    conf_thres: float = 0.25,
    show: bool = False,
    lane_config_json: Path | None = None,
    skip_frames: int = 0,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid_path = output_dir / "annotated.mp4"
    writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))

    # Load model & tracker
    model = load_detector(model_path)
    tracker = build_tracker()

    # Lanes
    if lane_config_json and lane_config_json.exists():
        lanes = json.loads(Path(lane_config_json).read_text())
    else:
        lanes = make_default_lanes(width, height)

    # Counts per lane and seen IDs per lane
    lane_counts = [0, 0, 0]
    lane_seen_ids = [set(), set(), set()]

    # CSV log rows
    event_rows = []  # dicts: {id, lane, frame, ts}

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if skip_frames and (frame_idx % (skip_frames + 1) != 1):
            continue

        # Detect
        results = model(frame, conf=conf_thres, verbose=False)[0]

        detections = []  # [[x1,y1,x2,y2], conf, class_name]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        clss = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []
        confs = results.boxes.conf.cpu().numpy() if results.boxes is not None else []

        for (x1, y1, x2, y2), c, cls in zip(boxes, confs, clss):
            if cls in VEHICLE_CLS:
                detections.append([[float(x1), float(y1), float(x2), float(y2)], float(c), VEHICLE_CLS[cls]])

        # Track
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw lanes
        for i, lane in enumerate(lanes, start=1):
            pts = np.array(lane, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(frame, f"Lane {i}", tuple(lane[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Handle tracks & counting
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            tid = trk.track_id
            ltrb = trk.to_ltrb()  # left, top, right, bottom
            x1, y1, x2, y2 = map(int, ltrb)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Which lane?
            lane_idx = None
            for i, lane in enumerate(lanes):
                if point_in_polygon((cx, cy), lane):
                    lane_idx = i
                    break

            # Draw track bbox & ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
            label = f"ID {tid}"
            cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)

            # Count only the first time this ID enters a lane
            if lane_idx is not None and tid not in lane_seen_ids[lane_idx]:
                lane_seen_ids[lane_idx].add(tid)
                lane_counts[lane_idx] += 1
                ts = frame_idx / fps
                event_rows.append({
                    "VehicleID": int(tid),
                    "LaneNumber": int(lane_idx + 1),
                    "FrameNumber": int(frame_idx),
                    "Timestamp": round(ts, 3),
                })

        # Overlay counts
        for i, c in enumerate(lane_counts, start=1):
            cv2.putText(
                frame,
                f"Lane {i}: {c}",
                (10, 30 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (50, 200, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        if show:
            cv2.imshow("Traffic Flow", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    # Save CSV & summary
    df = pd.DataFrame(event_rows, columns=["VehicleID", "LaneNumber", "FrameNumber", "Timestamp"])
    csv_path = output_dir / "events.csv"
    df.to_csv(csv_path, index=False)

    summary = {f"Lane {i+1}": int(c) for i, c in enumerate(lane_counts)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nProcessing complete.")
    print("Counts per lane:")
    for i, c in enumerate(lane_counts, start=1):
        print(f"  Lane {i}: {c}")
    print(f"CSV: {csv_path}")
    print(f"Video: {out_vid_path}")
    print(f"Summary: {(output_dir / 'summary.json')}")


# -------------------------
# CLI
# -------------------------




def main():

    print("Downloading video...")
    video_path = download_youtube("https://www.youtube.com/watch?v=MNn9qKG2UFI", "outputs")

    process_video(
    video_path = Path("outputs") / "4K Road traffic video for object detection and tracking - free download now!.mp4",
    output_dir=Path("outputs"),
    model_path="yolov8n.pt",
    conf_thres=0.25,
    show=False
)




if __name__ == "__main__":
    main()
