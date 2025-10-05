import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import pandas as pd, matplotlib.pyplot as plt
from FUCTIONS import *

# =========================
# Ρυθμίσεις / Παράμετροι
# =========================
W, H = 640, 480
CONF_THRES_DET = 0.25
CONF_THRES_POSE = 0.25
IOU_MATCH_THRES = 0.4
EPS = 1e-3

PRS_POSITION = 3
PRS_VEL = 2
PRS_KPS = 15
PRS_VEL_H = 15

POSE_EVERY = 3          # τρέξε pose κάθε 3 καρέ
FBF_MAX_AGE = 5

# =========================
# Ομογραφία (προς BEV σε μέτρα)
# =========================
# SOURCE = np.array([[257,132],[371,132],[516,480],[99,480]])  # videoAI1
# VIDEO_SOURSE = "../Tools_WorkSafety/Dataset-Videos/AI_video1.mp4"
# TARGET = np.array([[0,0],[6,0],[6,15],[0,15]])

SOURCE = np.array([[219,158],[389,158],[531,480],[107,480]])  # videoAI2
VIDEO_SOURSE = "../Tools_WorkSafety/Dataset-Videos/AI_video2.mp4"
TARGET = np.array([[0,0],[9,0],[9,18],[0,18]])

# SOURCE = np.array([[257,187],[378,187],[535,480],[106,480]])  # videoAI3
# VIDEO_SOURSE = "../Tools_WorkSafety/Dataset-Videos/AI_video3.mp4"
# TARGET = np.array([[0,0],[6,0],[6,15],[0,15]])

# SOURCE = np.array([[227,170],[420,170],[567,480],[171,480]])  # videoAI4
# VIDEO_SOURSE = "../Tools_WorkSafety/Dataset-Videos/AI_video4.mp4"
# TARGET = np.array([[0,0],[9,0],[6,17],[0,17]])

view_tf = ViewTransformer(source=SOURCE, target=TARGET)
number = 0
# =========================
# Loading YOLO Models
# =========================
model_det  = YOLO("../Tools_WorkSafety/Yolo-Weights/yolo_detAI_best.pt")
model_pose = YOLO("../Tools_WorkSafety/Yolo-Weights/yolo_poseAI_best.pt")
classNames = model_det.names

# =========================
# Dionionaries for each object id
# =========================
car_data = defaultdict(lambda: {
    "pos_px": [], "vel_px": [],
    "pos_m":  [], "vel_m":  [],
    "kps1":   [], "kps3":   [],
    "kps5": [] , "kps8": [],
    "bbox":   [], "fbf_dir_px": None,
    "fbf_age": 999
})
person_data = defaultdict(lambda: {
    "pos_px": [], "vel_px_H": [], "vel_px": [],
    "pos_m":  [], "vel_m_H":  [], "vel_m":  [],
    "distance" : [] , "bbox":   []
})

# Kalman dicts
kf_person_px, kf_car_px = {}, {}
kf_person_m,  kf_car_m  = {}, {}

risk_records = []
frame_count = 0

# =========================
# Loading the Video
# =========================
cap = cv2.VideoCapture(VIDEO_SOURSE)
fps = cap.get(cv2.CAP_PROP_FPS)  if cap.isOpened() else 24.0

while True:
    ok, img = cap.read()
    if not ok:
        break
    frame_count += 1
    frame = cv2.resize(img, (W, H))
    refreshed_ids = set()
    #     Tracker from Ultralytics
    #   - persist=True for stable IDs (the id number increase for every new detected object)
    #   - tracker='bytetrack.yaml' or tracker= 'botsort.yaml', device = 0 (for GPU)
    results_det = model_det.track(
        frame,
        imgsz=640, conf=CONF_THRES_DET, iou=0.35,
        tracker='bytetrack.yaml',
        persist=True,
        verbose=False
    )[0]
    print(f" FPS : {fps}")
    # if no ids founded continue
    if results_det.boxes is None or results_det.boxes.id is None:
        cv2.imshow("Tracking & Risk", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes_xyxy = results_det.boxes.xyxy.cpu().numpy()
    confs      = results_det.boxes.conf.cpu().numpy()
    clss       = results_det.boxes.cls.int().cpu().numpy()
    tids       = results_det.boxes.id.int().cpu().numpy()

    # for storing active ids in each frame
    active_car_ids    = set()
    active_person_ids = set()

    # ====== YOLO pose (1x /3 frame) ======
    pose_frame = (frame_count % POSE_EVERY == 0)

    if pose_frame:
        results_pose = model_pose(frame, imgsz = 480, conf=CONF_THRES_POSE, verbose=False)[0]
        pose_boxes = results_pose.boxes.xyxy.cpu().numpy() if results_pose.boxes is not None else np.empty((0, 4))
        pose_kps = results_pose.keypoints.xy.cpu().numpy() if (results_pose.keypoints is not None) else None
    else:
        results_pose = None
        pose_boxes = np.empty((0, 4))
        pose_kps = None

    # ====== Going through from all tracked detections ======
    for (x1, y1, x2, y2), conf, cls, tid in zip(boxes_xyxy, confs, clss, tids):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        name = classNames.get(int(cls), str(int(cls))).lower()

        # -------- Forklifts --------
        if name in ("forklift") or cls == 0:
            active_car_ids.add(int(tid))
            # cx, cy = int((x1 + x2)/2.0), int((y1+y2)/2)
            cx, cy = int((x1 + x2) / 2.0), int( y2)
            car_data[tid]["bbox"] = [x1, y1, x2, y2]

            # px
            car_data[tid]["pos_px"] = (car_data[tid]["pos_px"] + [(cx, cy)])[-PRS_POSITION:]
            # meters
            car_pos_m = view_tf.transform_points(np.array([cx, cy]))[0].astype(float)
            car_data[tid]["pos_m"]  = (car_data[tid]["pos_m"] + [car_pos_m])[-PRS_POSITION:]

            # KF velocities (dt=1/fps)
            v_px = compute_velocity_with_kalman1(kf_car_px, tid, np.array([cx, cy]), dt=1/fps,
                                                 sigma_meas=5.0 , sigma_accel=6.0)

            v_m  = compute_velocity_with_kalman1(kf_car_m,  tid, car_pos_m,          dt=1/fps,
                                                 sigma_meas=0.3, sigma_accel=3.0)

            car_data[tid]["vel_px"] = (car_data[tid]["vel_px"] + [v_px])[-PRS_VEL:]
            car_data[tid]["vel_m"]  = (car_data[tid]["vel_m"] + [v_m]) [-PRS_VEL:]

            # Matching pose→forklift with IoU
            if pose_boxes.size and (pose_kps is not None):

                best_iou, best_idx = -1.0, -1
                for i, pb in enumerate(pose_boxes):
                    j = iou_xyxy([x1,y1,x2,y2], pb)
                    if j > best_iou:
                        best_iou, best_idx = j, i

                if best_idx >= 0 and best_iou >= IOU_MATCH_THRES:
                    kps = pose_kps[best_idx]  # (K,2)

                    if kps.shape[0] > 8:
                        kp1 = tuple(map(int, kps[1]))
                        kp3 = tuple(map(int, kps[3]))
                        kp5 = tuple(map(int, kps[5]))
                        kp8 = tuple(map(int, kps[8]))

                        car_data[tid]["kps1"] = (car_data[tid]["kps1"] + [kp1])[-PRS_KPS:]
                        car_data[tid]["kps3"] = (car_data[tid]["kps3"] + [kp3])[-PRS_KPS:]
                        car_data[tid]["kps5"] = (car_data[tid]["kps5"] + [kp5])[-PRS_KPS:]
                        car_data[tid]["kps8"] = (car_data[tid]["kps8"] + [kp8])[-PRS_KPS:]

                        priority_pairs = [("kps3", "kps5"), ("kps1", "kps3"), ("kps3", "kps8"),
                                          ("kps1", "kps5"), ("kps5", "kps8"), ("kps1", "kps8")]
                        Fbf_dir_px = None
                        for ak, bk in priority_pairs:
                            if car_data[tid][ak] and car_data[tid][bk]:
                                a = np.array(car_data[tid][ak][-1], dtype=float)
                                b = np.array(car_data[tid][bk][-1], dtype=float)
                                vec = b - a
                                n = np.linalg.norm(vec)
                                if n >= 1.0:
                                    Fbf_dir_px = vec / n
                                    break

                        if Fbf_dir_px is not None:
                            car_data[tid]["fbf_dir_px"] = Fbf_dir_px
                            car_data[tid]["fbf_age"] = 0  # if new Fbf detected then refresh
                            refreshed_ids.add(tid)
            print(f"Car :{tid}-{3.6 * car_data[tid]["vel_m"][0]}")
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f'Forklift {tid}', (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            if car_data[tid]["vel_px"]:
                vx, vy = car_data[tid]["vel_px"][-1]
                cv2.arrowedLine(frame, (int(cx), int(cy)), (int(cx+0.5*vx), int(cy+0.5*vy)), (0,255,255), 2)

        # -------- Persons --------
        elif name in ("person") or cls == 1:
            active_person_ids.add(int(tid))
            # cx, cy = int((x1 + x2)/2.0), int((y1+y2)/2)
            cx, cy = int((x1 + x2) / 2.0), int(y2)
            person_data[tid]["bbox"] = [x1, y1, x2, y2]


            person_data[tid]["pos_px"] = (person_data[tid]["pos_px"] + [(cx, cy)])[-PRS_POSITION:]
            pos_m = view_tf.transform_points(np.array([cx, cy]))[0].astype(float)
            person_data[tid]["pos_m"]  = (person_data[tid]["pos_m"]  + [pos_m])[-PRS_POSITION:]

            # KF velocities (dt=1/fps)
            v_px = compute_velocity_with_kalman1(kf_person_px, tid, np.array([cx, cy]), dt=1/fps,
                                                 sigma_meas=10.0, sigma_accel=10.5)

            v_m  = compute_velocity_with_kalman1(kf_person_m,  tid, pos_m,              dt=1/fps,
                                                 sigma_meas = 0.2, sigma_accel=1.2)

            person_data[tid]["vel_px_H"] = (person_data[tid]["vel_px_H"] + [v_px])[-PRS_VEL_H:]
            person_data[tid]["vel_m_H"]  = (person_data[tid]["vel_m_H"]  + [v_m]) [-PRS_VEL_H:]

            if len( person_data[tid]["pos_px"]) >= PRS_POSITION  :
                pos_px_prev = np.array(person_data[tid]["pos_px"][-PRS_POSITION], dtype = float)
                pos_px_now = np.array( person_data[tid]["pos_px"][-1] ,dtype = float)
                distance = np.linalg.norm(pos_px_now - pos_px_prev)
            else :
                distance = 0.0

            person_data[tid]["distance"] = (person_data[tid]["distance"] + [distance])[-PRS_VEL_H:]
            # index = None
            if distance > 1 :
                index= -1
            else :
                index = None
                # idx_offset = None
                dist_hist = person_data[tid]["distance"]
                for i, d in enumerate(reversed(dist_hist), start=1):
                    if d > 5 :
                        index = -i
                        break
                if index is None:
                        index = -1  # fallback: τρέχουσα

            vel_px_hist = person_data[tid]["vel_px_H"]
            vel_m_hist = person_data[tid]["vel_m_H"]

            if vel_px_hist:
                j = index if -len(vel_px_hist) <= index < len(vel_px_hist) else -1
                velocity_px = vel_px_hist[j]
            else:
                velocity_px = v_px

            if vel_m_hist:
                j = index if -len(vel_m_hist) <= index < len(vel_m_hist) else -1
                velocity_m = vel_m_hist[j]
            else:
                velocity_m = v_m

            # --- 4) Storing Final Velocities Values ---
            person_data[tid]["vel_px"] = (person_data[tid]["vel_px"] + [velocity_px])[-PRS_VEL:]
            person_data[tid]["vel_m"] = (person_data[tid]["vel_m"] + [velocity_m])[-PRS_VEL:]

            # Optional : Draw Results
            print(f"Person : {tid}-{3.6*person_data[tid]["vel_m"][0]} - {distance}")
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, f'Person {tid}', (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            if person_data[tid]["vel_px"]:
                vx, vy = person_data[tid]["vel_px"][-1]
                cv2.arrowedLine(frame, (int(cx), int(cy)), (int(cx+0.5*vx), int(cy+0.5*vy)), (255,0,0), 2)

    # Cleaning Dictionaries from inactive ids
    for tid in list(car_data.keys()):
        if tid not in active_car_ids:
            del car_data[tid]
            if tid in kf_car_px: del kf_car_px[tid]
            if tid in kf_car_m:  del kf_car_m[tid]

    for tid, C in car_data.items():
        if (C["fbf_dir_px"] is not None) and (tid not in refreshed_ids):
            C["fbf_age"] += 1

    for tid in list(person_data.keys()):
        if tid not in active_person_ids:
            del person_data[tid]
            if tid in kf_person_px: del kf_person_px[tid]
            if tid in kf_person_m:  del kf_person_m[tid]



    # ====== Calculating fi for each pair (person/forklift) ======
    for pid, P in person_data.items():

        if len(P["pos_m"]) < PRS_POSITION or len(P["vel_m"]) < PRS_VEL:
            continue
        posP_m = np.array(P["pos_m"][-1], dtype=float)
        velP_m = np.mean(np.array(P["vel_m"][-2:]), axis=0)
        velP_px = np.mean(np.array(P["vel_px"][-2:]), axis=0)
        bboxP  = P["bbox"]
        posP_px = np.array(P["pos_px"][-1], dtype=float)

        # print(f"Pid: {pid} - PosPpx: {posP_px} - VelPpx: {velP_px}  -  PosPm: {posP_m} - VelPm: {3.6*velP_m}")

        for cid, C in car_data.items():
            if len(C["pos_m"]) < 1 or len(C["vel_m"]) < 1 or not C["bbox"]:
                continue
            posC_m = np.array(C["pos_m"][-1], dtype=float)
            velC_m = np.mean(np.array(C["vel_m"][-2:]),axis=0)
            bboxC  = C["bbox"]
            posC_px = np.array(C["pos_px"][-1], dtype=float)
            velC_px = np.mean(np.array(C["vel_px"][-2:]), axis=0)

            #  Checking if the detected person is a forklift driver with IoU
            if iou_xyxy(bboxP, bboxC) > 0.6:
                continue
            #  If the pair's distance is greater than 300 px , continue
            if np.linalg.norm(posP_px - posC_px) > 300:
                continue

            rel_pos_m = posP_m - posC_m   # from forklift view to worker location
            dist_now = np.linalg.norm(rel_pos_m)

            # If the pair's distance is greater than 12 meters , continue
            if dist_now > 12:
                continue

            rel_pos_px = posP_px - posC_px
            rel_vel_m = velP_m - velC_m
            P = np.dot(rel_vel_m,rel_pos_m)

            # Calculating TTC, TTCE, DCE
            ttce_m, dce_m = compute_ttce_dce(pos1=posP_m, vel1=velP_m, pos2=posC_m, vel2=velC_m)
            ttce_px, dce_px = compute_ttce_dce(pos1=posP_px, vel1=velP_px, pos2=posC_px, vel2=velC_px)
            ttc = ttc1(p1= posP_m, v1=velP_m, p2= posC_m, v2= velC_m, d = 1)


            Fbf_dir_px = car_data[cid]["fbf_dir_px"]
            if (Fbf_dir_px is None) or (car_data[cid]["fbf_age"] > FBF_MAX_AGE):
                # δεν έχουμε φρέσκια κατεύθυνση από pose -> προσπέρασε
                continue


            rel_dir = rel_pos_m / max(np.linalg.norm(rel_pos_m), EPS)
            vP_dir  = velP_m    / max(np.linalg.norm(velP_m),    EPS)

            Weights = [1, 2, 3, 6]

            ws = compute_W(rel_dir, vP_dir, Fbf_dir_px, weights= Weights)
            if not ws:
                continue
            w1, w2 = ws

            # print(f"Pid {pid} : {w1} - Cid {cid} : {w2}" )
            fi = risk_score(dist = dist_now, ttc = ttc, ttce = ttce_m, dce = dce_m , W1 = w1, W2= w2 )
            risk_records.append({
                "frame": frame_count,
                "time_sec": round(frame_count / max(fps, 1.0), 3),
                "person_id": pid,
                "car_id": cid,
                "fi": float(fi),
                "w1": int(w1),
                "w2": int(w2),
                "ttce": float(ttce_m),
                "dce": float(dce_m),
                "ttc": float(ttc),
                "dist_now": float(dist_now)
            })

            p1 = tuple(map(int, posP_px)); p2 = tuple(map(int, posC_px))
            color1 = (0,255,0)
            color2= (255,0,0)

            cv2.line(frame, p1, p2, color1, 2)
            mid = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
            cv2.putText(frame, f"TTC={ttc:.1f}s", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Optional : Print risk_records list
    # frame_rows = [r for r in risk_records if r["frame"] == frame_count]
    # if frame_rows:
    #     df_frame = pd.DataFrame(frame_rows)[["frame", "time_sec", "person_id", "car_id",
    #                                          "fi", "dist_now", "ttc", "ttce", "dce", "w1", "w2"]]
    #     df_frame = df_frame.rename(columns={"dist_now": "di", "ttc": "ttci", "ttce": "ttcei", "dce": "dcei"})
    #     df_frame = df_frame.sort_values(["person_id", "car_id"])
    #     print("\n--- Frame", frame_count, "---")
    #     print(df_frame.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    # Optional : Show the raw boxes/Kps from Yolo-Pose for checking
    # if pose_boxes is not None and pose_boxes.size > 0:
    #     for pb in pose_boxes:
    #         x1p, y1p, x2p, y2p = map(int, pb)
    #         cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (200, 200, 0), 1)

    if pose_kps is not None and pose_kps.size > 0:
        for kp in pose_kps:  # kp: (K, 2)
            for (x, y) in kp:
                cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1)

    # Optional : Show the homography points in the image for checking
    for index,point in enumerate(SOURCE):
        a = index
        x,y= int(point[0]), int(point[1])
        cv2.circle(frame, (x,y), 5, (255,255,255),-1)
        cv2.putText(frame, f"{a}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Tracking & Risk (ByteTrack)", frame)
    if cv2.waitKey(number) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()


# =====================================================================
# === Output directory ===
out_dir = "leading_indicators_results"
os.makedirs(out_dir, exist_ok=True)
df = pd.DataFrame(risk_records)
# df: columns -> frame, time_sec, person_id, car_id, fi, di, ttci, ttcei, dcei, w1, w2
df["pair"] = df["person_id"].astype(str) + "--" + df["car_id"].astype(str)

# ======= Diagrams =========
if df.empty:
    print("[risk] Δεν υπάρχουν records.")
else:
    # Pairing person-forklift ids
    df["pair"] = df["person_id"].astype(str) + "--" + df["car_id"].astype(str)

    # rename variables
    df = df.rename(columns={
        "fi": "fi",
        "dist_now": "di",
        "ttc": "ttci",
        "ttce": "ttcei",
        "dce": "dcei"
    })

    df["t"] = df["time_sec"].round(3)

    # In cace we have multiple lines with the same pair, we are calculating the mean values
    agg = (df.groupby(["t","pair"], as_index=False)
             [["fi","di","ttci","ttcei","dcei"]].mean())

    # ----- Sorting values for each forklift/person pair  -----
    fi_ts   = agg.pivot(index="t", columns="pair", values="fi").sort_index()
    di_ts   = agg.pivot(index="t", columns="pair", values="di").sort_index()
    ttci_ts = agg.pivot(index="t", columns="pair", values="ttci").sort_index()
    ttcei_ts= agg.pivot(index="t", columns="pair", values="ttcei").sort_index()
    dcei_ts = agg.pivot(index="t", columns="pair", values="dcei").sort_index()

    # ----- Value checking for ttc, ttce -----
    ttci_valid  = ttci_ts.mask(~np.isfinite(ttci_ts) | (ttci_ts <= 0))
    ttcei_valid = ttcei_ts.mask(~np.isfinite(ttcei_ts) | (ttcei_ts <= 0))

    # ----- Calculating the mean values -----
    f_mean    = fi_ts.mean(axis=1, skipna=True).rename("f_mean")
    d_mean    = di_ts.mean(axis=1, skipna=True).rename("d_mean")
    ttc_mean  = ttci_valid.mean(axis=1, skipna=True).rename("ttc_mean")
    ttce_mean = ttcei_valid.mean(axis=1, skipna=True).rename("ttce_mean")
    dce_mean  = dcei_ts.mean(axis=1, skipna=True).rename("dce_mean")

    # ----- Optional : Storing results in a CSV file -----
    fi_ts.to_csv(os.path.join(out_dir, "fi_timeseries_per_pair.csv"))
    di_ts.to_csv(os.path.join(out_dir, "di_timeseries_per_pair.csv"))
    ttci_ts.to_csv(os.path.join(out_dir, "ttci_timeseries_per_pair.csv"))
    ttcei_ts.to_csv(os.path.join(out_dir, "ttcei_timeseries_per_pair.csv"))
    dcei_ts.to_csv(os.path.join(out_dir, "dcei_timeseries_per_pair.csv"))

    means_df = pd.concat([f_mean, d_mean, ttc_mean, ttce_mean, dce_mean], axis=1)
    means_df.to_csv(os.path.join(out_dir, "means_per_time.csv"))

    print("[risk] Saved:")
    print("  - *_timeseries_per_pair.csv (ανά ζεύγος)")
    print("  - means_per_time.csv (μέσοι ανά frame)")

    # pair = fi_ts.columns[0]  # ή "3-7"
    # plt.figure(); plt.plot(fi_ts.index, fi_ts[pair]); plt.title(f"fi(t) — pair {pair}"); plt.xlabel("t [s]"); plt.ylabel("fi"); plt.show()

    # Showing Diagrams
    plt.figure(); plt.plot(means_df.index, means_df["f_mean"]); plt.title("f_mean(t)"); plt.xlabel("t [s]"); plt.ylabel("mean fi"); plt.show()
    plt.figure(); plt.plot(means_df.index, means_df["d_mean"]); plt.title("d_mean(t)"); plt.xlabel("t [s]"); plt.ylabel("mean distance"); plt.show()
    plt.figure(); plt.plot(means_df.index, means_df["ttc_mean"]); plt.title("ttc_mean(t)"); plt.xlabel("t [s]"); plt.ylabel("mean TTC"); plt.show()
    plt.figure(); plt.plot(means_df.index, means_df["ttce_mean"]); plt.title("ttce_mean(t)"); plt.xlabel("t [s]"); plt.ylabel("mean TTCE"); plt.show()
    plt.figure(); plt.plot(means_df.index, means_df["dce_mean"]); plt.title("dce_mean(t)"); plt.xlabel("t [s]"); plt.ylabel("mean DCE"); plt.show()
    plt.figure();  plt.plot(means_df.index , means_df["dce_mean"], label = "DCE")
    plt.plot(means_df.index , means_df["ttce_mean"], label = "TTCE")
    plt.title("TTCE & DCE vs t")
    plt.xlabel("t [s]")
    plt.ylabel("value")
    plt.legend()
    plt.show()
    W_CATS = [1, 2, 3, 6]

    # 1) Showing Histograms for w1
    w1_counts = (df["w1"].value_counts().reindex(W_CATS, fill_value=0)).astype(int)
    plt.figure()
    plt.bar(w1_counts.index.astype(str), w1_counts.values)
    plt.title("Histogram of w1")
    plt.xlabel("w1")
    plt.ylabel("count")
    plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, "hist_w1.png"), dpi=150)

    # 2) Showing Histogram for w2
    w2_counts = (df["w2"].value_counts().reindex(W_CATS, fill_value=0)).astype(int)
    plt.figure()
    plt.bar(w2_counts.index.astype(str), w2_counts.values)
    plt.title("Histogram of w2")
    plt.xlabel("w2")
    plt.ylabel("count")
    plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, "hist_w2.png"), dpi=150) # Optional : For storing results

    # 3) 2D Histogram for w1 , w2
    combo = (df.groupby(["w1", "w2"]).size()
             .unstack(fill_value=0)
             .reindex(index=W_CATS, columns=W_CATS, fill_value=0)
             )

    plt.figure()
    im = plt.imshow(combo.values, origin="lower", aspect="auto")
    plt.xticks(range(len(W_CATS)), W_CATS)
    plt.yticks(range(len(W_CATS)), W_CATS)
    plt.xlabel("w2");
    plt.ylabel("w1");
    plt.title("Counts of (w1, w2)")
    plt.colorbar(im, label="count")

    # Optional : Showing the values on each shell
    for i in range(combo.shape[0]):
        for j in range(combo.shape[1]):
            c = int(combo.iat[i, j])
            if c > 0:
                plt.text(j, i, str(c), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, "hist_w1_w2_matrix.png"), dpi=150)



    # ===== Constructing arrays for Critical Values =====

    fi_rows = []
    mins_rows = []

    for pair, g in df.groupby("pair"):
        pid = int(g["person_id"].iloc[0])
        cid = int(g["car_id"].iloc[0])

        # --- FI: max  for each pair ---
        idx_fi = g["fi"].idxmax()
        rfi = g.loc[idx_fi]
        fi_rows.append({
            "pair": pair,
            "person_id": pid, "car_id": cid,
            "fi_max": float(rfi["fi"]),
            "t_at_fi_max": float(rfi["time_sec"]),
            "frame_at_fi_max": int(rfi["frame"]),
            "w1_at_fi_max": int(rfi["w1"]),
            "w2_at_fi_max": int(rfi["w2"]),
            "di_at_fi_max": float(rfi["di"]),
            "ttci_at_fi_max": float(rfi["ttci"]),
            "ttcei_at_fi_max": float(rfi["ttcei"]),
            "dcei_at_fi_max": float(rfi["dcei"]),
        })

        # --- Ελάχιστα ---
        # Min Distance for each pair and the time it occured
        idx_di = g["di"].idxmin()
        rdi = g.loc[idx_di]

        # Min TTC for each pair and the time it occured
        g_ttc = g[np.isfinite(g["ttci"]) & (g["ttci"] > 0)]
        rttc = g_ttc.loc[g_ttc["ttci"].idxmin()] if not g_ttc.empty else None

        # Min TTCΕ for each pair and the time it occured
        g_ttce = g[np.isfinite(g["ttcei"]) & (g["ttcei"] > 0)]
        rttce = g_ttce.loc[g_ttce["ttcei"].idxmin()] if not g_ttce.empty else None

        # Min DCE for each pair and the time it occured
        g_dce = g[np.isfinite(g["dcei"])]
        rdce = g_dce.loc[g_dce["dcei"].idxmin()] if not g_dce.empty else None

        mins_rows.append({
            "pair": pair,
            "person_id": pid, "car_id": cid,

            "di_min": float(rdi["di"]),
            "t_at_di_min": float(rdi["time_sec"]),
            "frame_at_di_min": int(rdi["frame"]),

            "ttci_min": float(rttc["ttci"]) if rttc is not None else np.nan,
            "t_at_ttci_min": float(rttc["time_sec"]) if rttc is not None else np.nan,
            "frame_at_ttci_min": int(rttc["frame"]) if rttc is not None else -1,


            "dcei_min": float(rdce["dcei"]) if rdce is not None else np.nan,
            "ttcei_at_dcei_min": float(rdce["ttcei"]) if rdce is not None else np.nan,
            "t_at_dcei_min": float(rdce["time_sec"]) if rdce is not None else np.nan,
            "frame_at_dcei_min": int(rdce["frame"]) if rdce is not None else -1,
        })

    fi_crit_df = pd.DataFrame(fi_rows).sort_values(["person_id", "car_id"])
    mins_crit_df = pd.DataFrame(mins_rows).sort_values(["person_id", "car_id"])


    # --- Rounding Values ---
    def round_cols(df_, cols):
        for c in cols:
            if c in df_.columns:
                df_[c] = df_[c].round(2)


    round_cols(fi_crit_df,
               ["fi_max", "t_at_fi_max", "di_at_fi_max", "ttci_at_fi_max", "ttcei_at_fi_max", "dcei_at_fi_max"])
    round_cols(mins_crit_df,
               ["di_min", "t_at_di_min", "ttci_min", "t_at_ttci_min", "ttcei_min", "t_at_ttcei_min", "dcei_min",
                "t_at_dcei_min"])

    # --- Saving the arrays in "output_crit_value" ---
    crit_dir = os.path.join(out_dir, "outputs_crit_value")
    os.makedirs(crit_dir, exist_ok=True)
    fi_path = os.path.join(crit_dir, "fi_max_per_pair.csv")
    mins_path = os.path.join(crit_dir, "mins_per_pair.csv")

    fi_crit_df.to_csv(fi_path, index=False, encoding="utf-8-sig")
    mins_crit_df.to_csv(mins_path, index=False, encoding="utf-8-sig")

    print(f"[risk] Saved critical values to: {crit_dir}")
