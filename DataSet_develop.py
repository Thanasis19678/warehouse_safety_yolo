import os
import cv2
import random
from ultralytics import YOLO
from pathlib import Path

# ============================
# Inputs
# ============================
VIDEO_INPUT = r"../Tools_WorkSafety/Dataset-Videos"  # folder or file (mp4/mkv/avi/mov...)
OUTPUT_DET = "dataset_detection"
OUTPUT_POSE = "dataset_pose"

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".mpg", ".m4v", ".wmv")

# ============================
# Models
# ============================
model_det  = YOLO("../Tools_WorkSafety/Yolo-Weights/yolo-det-best.pt")   # detection
model_pose = YOLO("../Tools_WorkSafety/Yolo-Weights/yolo-pose-best.pt")  # pose

# =============================
# Parameters
# =============================
FRAME_STEP = 2       # κάθε πόσα frames κρατάμε
VAL_SPLIT  = 0.2     # validation percentage
IMG_SIZE   = (640, 480)

# ==============================
# Prepare output folders
# ==============================
for d in [f"{OUTPUT_DET}/images/train", f"{OUTPUT_DET}/images/val",
          f"{OUTPUT_DET}/labels/train", f"{OUTPUT_DET}/labels/val",
          f"{OUTPUT_POSE}/images/train", f"{OUTPUT_POSE}/images/val",
          f"{OUTPUT_POSE}/labels/train", f"{OUTPUT_POSE}/labels/val"]:
    os.makedirs(d, exist_ok=True)

# ============================
# Utilities
# ============================
def iter_video_paths(inp):
    """folder or file processing."""
    p = Path(inp)
    if p.is_dir():
        files = []
        for ext in VIDEO_EXTS:
            files.extend(p.glob(f"*{ext}"))
        return sorted(str(f) for f in files)
    elif p.is_file():
        return [str(p)]
    else:
        raise FileNotFoundError(f"VIDEO_INPUT not found: {p}")

def save_yolo_det(label_path, boxes_obj, img_w, img_h):
    """YOLO detection label per frame."""
    with open(label_path, "w") as f:
        if boxes_obj is None or len(boxes_obj) == 0:
            return
        xyxy = boxes_obj.xyxy.cpu().numpy()
        clss = boxes_obj.cls.cpu().numpy().astype(int)
        for cls, (x1, y1, x2, y2) in zip(clss, xyxy):
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def save_yolo_pose(label_path, bbox, keypoints, img_w, img_h, cls_id=0):
    """
    YOLO pose format: class cx cy w h followed by (x y v) * K
    keypoints: list of (x, y, conf) σε pixels.
    """
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    parts = [f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"]
    for (x, y, c) in keypoints:
        if c is None or c < 0.4:
            vx = vy = 0.0
            v = 0
        else:
            vx = x / img_w
            vy = y / img_h
            v = 2  # ορατό
        parts.append(f"{vx:.6f} {vy:.6f} {v}")

    with open(label_path, "a") as f:
        f.write(" ".join(parts) + "\n")

# ============================
# Main
# ============================
random.seed(42)
video_paths = iter_video_paths(VIDEO_INPUT)

for video in video_paths:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[warn] Cannot open video: {video}")
        continue

    frame_idx = 0
    video_name = Path(video).stem
    print(f"[info] Processing: {video_name}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % FRAME_STEP != 0:
            frame_idx += 1
            continue

        frame = cv2.resize(frame, IMG_SIZE)
        h, w, _ = frame.shape

        # ---------- Detection ----------
        det_res = model_det(frame, imgsz=IMG_SIZE[0], conf=0.25, verbose=False)[0]
        if det_res.boxes is not None and len(det_res.boxes) > 0:
            subset = "val" if random.random() < VAL_SPLIT else "train"
            img_name = f"{video_name}_{frame_idx}.jpg"

            # image detection
            img_path_det = f"{OUTPUT_DET}/images/{subset}/{img_name}"
            cv2.imwrite(img_path_det, frame)

            # labels detection
            label_path_det = f"{OUTPUT_DET}/labels/{subset}/{Path(img_name).stem}.txt"
            save_yolo_det(label_path_det, det_res.boxes, w, h)

            # ---------- Pose  ----------
            pose_res = model_pose(frame, imgsz=IMG_SIZE[0], conf=0.25, verbose=False)[0]
            img_path_pose  = f"{OUTPUT_POSE}/images/{subset}/{img_name}"
            label_path_pose = f"{OUTPUT_POSE}/labels/{subset}/{Path(img_name).stem}.txt"
            cv2.imwrite(img_path_pose, frame)

            if pose_res.boxes is not None and pose_res.keypoints is not None and len(pose_res.boxes) > 0:
                kps_xy   = pose_res.keypoints.xy.cpu().numpy()   # (N, K, 2)
                kps_conf = pose_res.keypoints.conf.cpu().numpy() # (N, K)
                boxes_xy = pose_res.boxes.xyxy.cpu().numpy()     # (N, 4)
                clss     = pose_res.boxes.cls
                if clss is not None:
                    clss = clss.cpu().numpy().astype(int)
                else:
                    clss = [0] * len(boxes_xy)

                for i, (kxy, kc, bb) in enumerate(zip(kps_xy, kps_conf, boxes_xy)):
                    keypoints = [(float(x), float(y), float(c)) for (x, y), c in zip(kxy, kc)]
                    save_yolo_pose(label_path_pose, bb, keypoints, w, h, cls_id=int(clss[i]))
            else:

                open(label_path_pose, "w").close()

        frame_idx += 1

    cap.release()

print("✅ Dataset έτοιμο για YOLO-Detection & YOLO-Pose εκπαίδευση")


