import os, json, cv2, shutil
from pathlib import Path

# ==========================
# Paths
# ==========================
DATASET_DIR = "../Tools_WorkSafety/dataset_pose"
EXPORT_DIR  = "dataset_pose_coco"
OUTPUT_JSON = os.path.join(EXPORT_DIR, "annotations_coco.json")

# ==========================
# Config
# ==========================
NUM_KEYPOINTS = 13
KEYPOINT_NAMES = [f"kp{i}" for i in range(1, NUM_KEYPOINTS+1)]

# Pose-Classes
CLASS_MAP = {
    0: "forklift"
}
# build cat list & inverse map
CATEGORIES = [{"id": cid+1, "name": name, "keypoints": KEYPOINT_NAMES, "skeleton": []}
              for cid, (_, name) in enumerate(sorted(CLASS_MAP.items(), key=lambda x: x[0]))]
CLS_TO_CATID = {cls: i+1 for i, cls in enumerate(sorted(CLASS_MAP.keys()))}

# ==========================
# Helpers fuctions
# ==========================
def yolo_to_coco_bbox(xc, yc, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    x1 = (xc * img_w) - bw/2.0
    y1 = (yc * img_h) - bh/2.0
    return [float(x1), float(y1), float(bw), float(bh)]

def fix_kps_length(kps_values, K):
    """Επιστρέφει λίστα μήκους 3*K (pad με μηδενικά ή truncate αν χρειάζεται)."""
    need = 3*K
    vals = list(kps_values)
    if len(vals) < need:
        vals += [0.0] * (need - len(vals))
    elif len(vals) > need:
        vals = vals[:need]
    return vals

def yolo_to_coco_keypoints(values, img_w, img_h):
    kps = []
    for i in range(0, len(values), 3):
        x, y, v = values[i:i+3]
        v = int(v)  # 0/1/2
        if v <= 0:
            kps.extend([0, 0, 0])
        else:
            kps.extend([float(x)*img_w, float(y)*img_h, v])
    return kps

# ==========================
# Prepare Export Dir
# ==========================
if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)
os.makedirs(os.path.join(EXPORT_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(EXPORT_DIR, "images", "val"),   exist_ok=True)

# ==========================
# Conversion
# ==========================
images = []
annotations = []
ann_id = 1

for split in ["train", "val"]:
    image_dir = Path(DATASET_DIR) / "images" / split
    label_dir = Path(DATASET_DIR) / "labels" / split

    img_paths = []
    for ext in ("*.jpg","*.jpeg","*.png"):
        img_paths.extend(image_dir.glob(ext))

    for img_file in sorted(img_paths):
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"[warn] cannot read {img_file}")
            continue
        h, w = img.shape[:2]

        # copy image to export
        export_img_path = Path(EXPORT_DIR) / "images" / split / img_file.name
        shutil.copy(img_file, export_img_path)

        img_id = len(images) + 1
        images.append({
            "id": img_id,
            "file_name": f"images/{split}/{img_file.name}",
            "height": h,
            "width":  w
        })

        # labels
        label_file = label_dir / (img_file.stem + ".txt")
        if not label_file.exists():
            # keep image with no annotations (negative sample)
            continue

        with open(label_file, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
            except:
                continue
            floats = list(map(float, parts[1:]))

            if len(floats) < 4:
                continue
            xc, yc, bw, bh = floats[:4]
            kps_values = floats[4:]

            # enforce length 3 * NUM_KEYPOINTS
            kps_values = fix_kps_length(kps_values, NUM_KEYPOINTS)

            bbox = yolo_to_coco_bbox(xc, yc, bw, bh, w, h)
            keypoints = yolo_to_coco_keypoints(kps_values, w, h)
            num_kps   = sum(1 for v in keypoints[2::3] if v > 0)

            cat_id = CLS_TO_CATID.get(cls, 1)  # default 1 αν λείπει mapping

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": int(num_kps)
            })
            ann_id += 1

# ==========================
# COCO JSON
# ==========================
coco = {
    "images": images,
    "annotations": annotations,
    "categories": CATEGORIES
}
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2, ensure_ascii=False)

print(f"✅ Έτοιμο COCO dataset στο: {EXPORT_DIR}\n  images: {len(images)} | anns: {len(annotations)}")

