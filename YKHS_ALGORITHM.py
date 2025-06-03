
from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import csv
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import seaborn as sns


# ----------------------------
# CSV Initialization
# ----------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file_path = f"iou_tracking_data_{timestamp}.csv"

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Frame", "Track_ID", "Detection_Index", "Label",
        "YOLO_CX", "YOLO_CY", "YOLO_W", "YOLO_H",
        "Kalman_CX", "Kalman_CY", "Kalman_W", "Kalman_H",
        "IoU", "Kx", "Ky",
        "Corrected_CX", "Corrected_CY", "Corrected_W", "Corrected_H",
        "Fused_CX", "Fused_CY", "Fused_W", "Fused_H",
    ])


# ----------------------------
# Tracking Utilities
# ----------------------------
class Track:
    count = 0
    def __init__(self, bbox, label):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R *= 10
        self.kf.P *= 3
        self.kf.Q *= 1
        x, y, w, h = bbox
        self.kf.x[:4] = np.array([[x], [y], [w], [h]])
        self.id = Track.count
        Track.count += 1
        self.skipped = 0
        self.history = []
        self.last_position = None
        self.speed = 0
        self.label = label
        self.predicted_positions = []
        self.corrected_positions = []
        self.measurements = []
        self.k_gain = (0.0, 0.0)

    def predict(self):
        self.kf.predict()
        pred = self.kf.x[:4].flatten()
        self.predicted_positions.append(pred.copy())
        return pred

    def update(self, bbox):
        self.measurements.append(bbox.copy())
        self.kf.update(bbox.reshape((4, 1)))
        self.corrected_positions.append(self.kf.x[:4].flatten().copy())
        self.k_gain = (float(self.kf.K[0, 0]), float(self.kf.K[1, 1]))
        self.skipped = 0
        cx, cy = bbox[0], bbox[1]
        self.history.append((int(cx), int(cy)))
        if self.last_position:
            dx = cx - self.last_position[0]
            dy = cy - self.last_position[1]
            self.speed = np.sqrt(dx ** 2 + dy ** 2)
        else:
            self.speed = 0
        self.last_position = (cx, cy)

def iou(bb1, bb2):
    x1, y1, x2, y2 = bb1
    x3, y3, x4, y4 = bb2
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bb1_area = (x2 - x1) * (y2 - y1)
    bb2_area = (x4 - x3) * (y4 - y3)
    union = bb1_area + bb2_area - inter_area
    return inter_area / union if union > 0 else 0

def draw_visuals(frame, track):
    # 🔴 Kalman corrected box
    x, y, w, h = track.kf.x[:4]
    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, f"ID {track.id} | {track.label} | {track.speed:.1f}px/f", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 🔵 Kalman prediction
    if track.predicted_positions:
        x, y, w, h = track.predicted_positions[-1]
        px1, py1, px2, py2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 1)

    # 🟢 YOLO detection
    if track.measurements:
        x, y, w, h = track.measurements[-1]
        dx1, dy1, dx2, dy2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 1)

    if len(track.history) >= 2:
        for i in range(1, len(track.history)):
            cv2.line(frame, track.history[i - 1], track.history[i], (0, 255, 255), 1)

# ----------------------------
# Main Loop
# ----------------------------
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("football.mov")
# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define output filename
output_filename = f"tracked_output_{timestamp}.mp4"

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or use 'XVID' for .avi
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

if not cap.isOpened():
    print("❌ Could not open video.")
else:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✅ Total number of frames: {total_frames}")
tracks = []
iou_threshold = 0.3
max_skipped = 10
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, stream=True)
    detections = []
    labels = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            detections.append(np.array([cx, cy, w, h]))
            labels.append(model.names[int(box.cls[0])])

    predictions = [track.predict() for track in tracks]
    matched, unmatched_dets, unmatched_trks = [], list(range(len(detections))), list(range(len(tracks)))

    if predictions and detections:
        cost_matrix = np.zeros((len(predictions), len(detections)))
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections):
                box1 = [pred[0] - pred[2] / 2, pred[1] - pred[3] / 2, pred[0] + pred[2] / 2, pred[1] + pred[3] / 2]
                box2 = [det[0] - det[2] / 2, det[1] - det[3] / 2, det[0] + det[2] / 2, det[1] + det[3] / 2]
                iou_val = iou(box1, box2)
                cost_matrix[i][j] = 1 - iou_val

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched = [(r, c) for r, c in zip(row_ind, col_ind) if cost_matrix[r][c] < 1 - iou_threshold]
        unmatched_trks = [i for i in range(len(tracks)) if i not in [m[0] for m in matched]]
        unmatched_dets = [i for i in range(len(detections)) if i not in [m[1] for m in matched]]

        for t, d in matched:
            tracks[t].update(detections[d])
            tracks[t].label = labels[d]
            pred = predictions[t]
            det = detections[d]
            iou_val = 1 - cost_matrix[t][d]
            alpha = 0.5
            fused = [alpha * p + (1 - alpha) * q for p, q in zip(pred, det)]

            with open(csv_file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    frame_count, t, d, labels[d],
                    round(det[0], 2), round(det[1], 2), round(det[2], 2), round(det[3], 2),
                    round(pred[0], 2), round(pred[1], 2), round(pred[2], 2), round(pred[3], 2),
                    round(iou_val, 4),
                    round(tracks[t].k_gain[0], 4), round(tracks[t].k_gain[1], 4),
                    round(fused[0], 2), round(fused[1], 2), round(fused[2], 2), round(fused[3], 2),
                    round(tracks[t].kf.x[0, 0], 2), round(tracks[t].kf.x[1, 0], 2),
                    round(tracks[t].kf.x[2, 0], 2), round(tracks[t].kf.x[3, 0], 2)
                ])

    for idx in unmatched_dets:
        tracks.append(Track(detections[idx], labels[idx]))

    for t in reversed(range(len(tracks))):
        tracks[t].skipped += 1
        if tracks[t].skipped > max_skipped:
            tracks.pop(t)

    class_counts = defaultdict(int)
    for track in tracks:
        draw_visuals(frame, track)
        class_counts[track.label] += 1

    y_offset = 30
    for obj_class, count in class_counts.items():
        cv2.putText(frame, f"{obj_class}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    # Save frame to output video
    out.write(frame)
    # 🔘 PAUSE / RESUME / QUIT CONTROLS
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        print("⏸ Paused... Press 'r' to resume or 'q' to quit.")
        while True:
            pause_key = cv2.waitKey(0) & 0xFF
            if pause_key == ord('r'):
                print("▶️ Resumed.")
                break
            elif pause_key == ord('q'):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                exit()

    # Display frame
    cv2.imshow("YOLO+Kalman+Hungarian", frame)


import glob

# Find the most recent iou_tracking_data_*.csv file
list_of_files = glob.glob("iou_tracking_data_*.csv")
if not list_of_files:
    raise FileNotFoundError("No tracking CSV files found.")

# Sort by creation/modification time
latest_csv_file = max(list_of_files, key=os.path.getmtime)

# Load the latest CSV
df = pd.read_csv(latest_csv_file)
print(f"✅ Loaded latest CSV file: {latest_csv_file}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")


if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)

    if not df.empty and all(col in df.columns for col in [
        "YOLO_CX", "YOLO_CY", "YOLO_W", "YOLO_H",
        "Kalman_CX", "Kalman_CY", "Kalman_W", "Kalman_H"
    ]):
        mse_cx = ((df["YOLO_CX"] - df["Kalman_CX"]) ** 2).mean()
        mse_cy = ((df["YOLO_CY"] - df["Kalman_CY"]) ** 2).mean()
        mse_w  = ((df["YOLO_W"]  - df["Kalman_W"])  ** 2).mean()
        mse_h  = ((df["YOLO_H"]  - df["Kalman_H"])  ** 2).mean()

        image_width = 1920
        image_height = 1080

        norm_mse_cx = mse_cx / (image_width ** 2)
        norm_mse_cy = mse_cy / (image_height ** 2)
        norm_mse_w  = mse_w  / (image_width ** 2)
        norm_mse_h  = mse_h  / (image_height ** 2)
        norm_overall_mse = (norm_mse_cx + norm_mse_cy + norm_mse_w + norm_mse_h) / 4

        print("\n====== Normalized MSE Summary ======")
        print(f"MSE (Center X):     {norm_mse_cx:.6f}")
        print(f"MSE (Center Y):     {norm_mse_cy:.6f}")
        print(f"MSE (Width):        {norm_mse_w:.6f}")
        print(f"MSE (Height):       {norm_mse_h:.6f}")
        print(f"Overall Normalized MSE: {norm_overall_mse:.6f}")
    else:
        print("⚠️ CSV file is empty or missing required columns.")
else:
    print("⚠️ CSV file not found. It seems no data was written during tracking.")



# Only continue if enough data exists
if len(df) >= 2:
    frames = range(len(df))

    # Create a 3x1 subplot layout
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    #---------------------------------------------------------------------------------------
    #Plot CX
    # Plot CX comparison on the first subplot
    axs[0].plot(frames, df["YOLO_CX"], label="YOLO_CX (Measurement)", color='green')
    axs[0].set_title("Center X Over Time for YOLO_CX (Measurement)")
    axs[0].set_xlabel("Frame Index")
    axs[0].set_ylabel("Center X (pixels)")
    axs[0].grid(True)

    # Plot CX comparison on the second subplot
    axs[1].plot(frames, df["Kalman_CX"], label="Kalman_CX (Prediction)", color='orange')
    axs[1].set_title("Center X Over Time for Kalman_CX (Prediction)")
    axs[1].set_xlabel("Frame Index")
    axs[1].set_ylabel("Center X (pixels)")
    axs[1].grid(True)

    # Plot CX comparison on the Third subplot
    axs[2].plot(frames, df["YOLO_CX"], label="YOLO_CX (Measurement)", color='green')
    axs[2].plot(frames, df["Kalman_CX"], label="Kalman_CX (Prediction)", color='orange')
    axs[2].set_title("Center X Over Time for Comparing both YOLO_CX (Measurement) & Kalman_CX (Prediction)")
    axs[2].set_xlabel("Frame Index")
    axs[2].set_ylabel("Center X (pixels)")
    axs[2].legend()
    axs[2].grid(True)


    # Adjust layout
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Not enough data to plot predicted vs. measured values.")

    # ---------------------------------------------------------------------------------------
    # Only continue if enough data exists
if len(df) >= 2:
    frames = range(len(df))

    # Create a 3x1 subplot layout
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    # Plot CY
    # Plot CY comparison on the first subplot
    axs[0].plot(frames, df["YOLO_CY"], label="YOLO_CY (Measurement)", color='green')
    axs[0].set_title("Center Y Over Time YOLO_CY (Measurement)")
    axs[0].set_xlabel("Frame Index")
    axs[0].set_ylabel("Center Y (pixels)")
    # axs[0].legend()
    axs[0].grid(True)

    # Plot CY comparison on the second subplot
    axs[1].plot(frames, df["Kalman_CY"], label="Kalman_CY (Prediction)", color='orange')
    axs[1].set_title("Center Y Over Time for Kalman_CY (Prediction)")
    axs[1].set_xlabel("Frame Index")
    axs[1].set_ylabel("Center Y (pixels)")
    axs[1].grid(True)

    # Plot CY comparison on the Third subplot
    axs[2].plot(frames, df["YOLO_CY"], label="YOLO_CY (Measurement)", color='green')
    axs[2].plot(frames, df["Kalman_CY"], label="Kalman_CY (Prediction)", color='orange')
    axs[2].set_title("Center Y Over Time for Comparing both YOLO_CY (Measurement) & Kalman_CY (Prediction)")
    axs[2].set_xlabel("Frame Index")
    axs[2].set_ylabel("Center Y (pixels)")
    axs[2].legend()
    axs[2].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

else:
    print("⚠️ Not enough data to plot predicted vs. measured values.")


# Select columns of interest
cols_of_interest = [
    "YOLO_CX", "Kalman_CX", "Corrected_CX", "Fused_CX",
    "YOLO_CY", "Kalman_CY", "Corrected_CY", "Fused_CY",
    "YOLO_W",  "Kalman_W",  "Corrected_W",  "Fused_W",
    "YOLO_H",  "Kalman_H",  "Corrected_H",  "Fused_H"
]

# Ensure all columns exist
existing_cols = [col for col in cols_of_interest if col in df.columns]

# Compute correlation matrix
correlation_matrix = df[existing_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson Correlation Between YOLO, Kalman, Corrected and Fused Variables")
plt.tight_layout()
plt.show()

if "IoU" in df.columns:
    average_iou = df["IoU"].mean()
    print(f"📊 Average IoU: {average_iou:.4f}")
else:
    print("⚠️ 'IoU' column not found in the DataFrame.")






