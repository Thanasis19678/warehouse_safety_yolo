import cv2
from ultralytics import YOLO
import numpy as np

# Φόρτωσε το YOLO-Pose μοντέλο σου
model = YOLO("../Tools_WorkSafety/Yolo-Weights/yolo_poseAI_best.pt")
model1 = YOLO("../Tools_WorkSafety/Yolo-Weights/yolo_detAI_best.pt")

# Βίντεο είσοδος
video_path = "../Tools_WorkSafety/Dataset-Videos/AI_video4.mp4"   # <-- βάλε το δικό σου
cap = cv2.VideoCapture(video_path)

# Αποθήκευση αποτελεσμάτων σε νέο βίντεο (προαιρετικό)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    # Τρέξε YOLO-Pose
    results = model(frame, conf=0.25, verbose=False)[0]
    results1 = model1(frame, conf=0.25, verbose=False)[0]
    # Σχεδίαση πάνω στο frame
    # annotated_frame = results.plot()  # αυτόματα ζωγραφίζει boxes + keypoints

    pose_boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.empty((0, 4))
    pose_kps = results.keypoints.xy.cpu().numpy() if (results.keypoints is not None) else None

    boxes_xyxy = results1.boxes.xyxy.cpu().numpy()
    confs = results1.boxes.conf.cpu().numpy()
    clss = results1.boxes.cls.int().cpu().numpy()

    for (x1, y1, x2, y2) , conf,cls in zip(boxes_xyxy, confs, clss) :

        x1 , y1, x2, y2 = int(x1) , int(y1), int(x2), int(y2)

        cx , cy  = int((x1 + x2) / 2.0), int(( y1 +y2)/2)

        if cls == 0 :

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{cls}', (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else :
            continue
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.putText(frame, f'{cls}', (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # cv2.arrowedLine(frame, (0,0), (cx, cy),(0,0,255), 2, 1)

    if pose_boxes is not None and pose_boxes.size > 0:
        for pb in pose_boxes:
            x1p, y1p, x2p, y2p = map(int, pb)
            cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (200, 200, 0), 2)

    else:
        continue
    #
    # if  pose_kps is not None and pose_kps.size > 0:
    #     for  kp in pose_kps:  # kp: (K, 2)
    #         for i, (x, y) in  enumerate(kp):
    #             cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1)
    #             # print (i)
    #             cv2.putText(frame, f'{i}', (int(x) +5 , int(y) +8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)
    # else :
    #     continue
    cv2.imshow("Tracking & Risk (ByteTrack)", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
      break


cap.release()
cv2.destroyAllWindows()




















