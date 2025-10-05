import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

Video_Path = r"../Tools_WorkSafety/Dataset-Videos/AI_video4.mp4"   # Εδώ εισάγεται το βίντεο ή η φωτογραφία της αποθήκης

def is_video(path):
    return str(path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"))

def read_first_frame(path):
    path = str(path)
    if is_video(path):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Couldn't read first frame from video.")
        return frame
    else:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError("Couldn't read image file.")
        return img

def main():
    parser = argparse.ArgumentParser(description="Drag 4 points over a 640x480 image inside a 1000x1000 canvas and print their image-relative coordinates.")
    parser.add_argument("input_path", nargs="?", default=None, help="Path to a video or image. If omitted, a blank 640x480 will be used.")
    parser.add_argument("--save-json", default=None, help="Optional path to save points as JSON.")
    args = parser.parse_args()
    args.input_path = Video_Path


    # Load first frame or create blank
    if args.input_path is None:
        frame0 = np.full((480, 640, 3), 30, dtype=np.uint8)  # dark gray placeholder
        cv2.putText(frame0, "No input_path provided", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
    else:
        frame_raw = read_first_frame(args.input_path)
        frame0 = cv2.resize(frame_raw, (640, 480), interpolation=cv2.INTER_AREA)

    # Canvas 1000x1000 with image centered
    CANVAS_W ,  CANVAS_H = 980 , 1000
    IMG_W, IMG_H = 640, 480
    OFFSET_X = (CANVAS_W - IMG_W) // 2  # 180
    OFFSET_Y = 50  # 260

    # Compose base canvas with the image placed at (OFFSET_X, OFFSET_Y)
    canvas_base = np.full((CANVAS_H, CANVAS_W, 3), 20, dtype=np.uint8)
    canvas_base[OFFSET_Y:OFFSET_Y+IMG_H, OFFSET_X:OFFSET_X+IMG_W] = frame0

    # Initial 4 points in IMAGE coords (TL, TR, BR, BL)
    pts_img = np.array([[0,0],[IMG_W,0],[IMG_W,IMG_H],[0,IMG_H]], dtype=np.float32)
    # Convert to CANVAS coords to draw
    pts_canvas = pts_img + np.array([OFFSET_X, OFFSET_Y], dtype=np.float32)

    # Dragging state
    RADIUS = 8
    dragging_idx = None

    win_name = "Quad Picker (1000x1000 canvas)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, CANVAS_W, CANVAS_H)

    def on_mouse(event, x, y, flags, userdata):
        nonlocal dragging_idx, pts_canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            # pick nearest point if within radius
            dists = np.linalg.norm(pts_canvas - np.array([x, y], dtype=np.float32), axis=1)
            j = int(np.argmin(dists))
            if dists[j] <= RADIUS * 1.6:
                dragging_idx = j
        elif event == cv2.EVENT_MOUSEMOVE and dragging_idx is not None:
            nx = np.clip(x, 0, CANVAS_W-1)
            ny = np.clip(y, 0, CANVAS_H-1)
            pts_canvas[dragging_idx] = [nx, ny]
        elif event == cv2.EVENT_LBUTTONUP:
            dragging_idx = None

    cv2.setMouseCallback(win_name, on_mouse)

    print("\nInstructions:")
    print(" - Drag the 4 points (1..4) with the mouse.")
    print(" - Press 'P' to print the image-relative coordinates (can be negative or > 640/480).")
    print(" - Press 'S' to save them (and optionally JSON if --save-json given).")
    print(" - Press 'R' to reset to image corners.")
    print(" - Press 'Q' or ESC to quit.\n")

    def draw_canvas():
        vis = canvas_base.copy()
        # draw quadrilateral lines
        for i in range(4):
            p1 = tuple(np.int32(pts_canvas[i]))
            p2 = tuple(np.int32(pts_canvas[(i+1)%4]))
            cv2.line(vis, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        # draw handles and labels 1..4 with coordinates in IMAGE space
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, p in enumerate(pts_canvas):
            px, py = int(p[0]), int(p[1])
            cv2.circle(vis, (px, py), RADIUS, (0, 0, 255), -1, cv2.LINE_AA)
            # label number
            cv2.putText(vis, str(i+1), (px + 10, py - 10), font, 0.7, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, str(i+1), (px + 10, py - 10), font, 0.7, (255,255,255), 1, cv2.LINE_AA)
            # show image-relative coords next to it
            ix = int(round(px - OFFSET_X))
            iy = int(round(py - OFFSET_Y))
            txt = f"({ix},{iy})"
            cv2.putText(vis, txt, (px + 10, py + 18), font, 0.55, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, txt, (px + 10, py + 18), font, 0.55, (255,255,255), 1, cv2.LINE_AA)

        # HUD: boundaries
        cv2.rectangle(vis, (OFFSET_X, OFFSET_Y), (OFFSET_X+IMG_W-1, OFFSET_Y+IMG_H-1), (120, 200, 120), 2, cv2.LINE_AA)
        cv2.putText(vis, "Image 640x480 inside 1000x1000 canvas", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, "Image 640x480 inside 1000x1000 canvas", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 1, cv2.LINE_AA)
        cv2.putText(vis, "Keys: P=print, S=save, R=reset, Q/Esc=quit", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, "Keys: P=print, S=save, R=reset, Q/Esc=quit", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
        return vis

    def get_points_image_coords():
        # convert current canvas points to image-relative coords (can be negative or > 640/480)
        pts_img_now = pts_canvas - np.array([OFFSET_X, OFFSET_Y], dtype=np.float32)
        return np.round(pts_img_now).astype(int)

    while True:
        vis = draw_canvas()
        cv2.imshow(win_name, vis)
        key = cv2.waitKey(16) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('r'):
            pts_img = np.array([[0,0],[IMG_W,0],[IMG_W,IMG_H],[0,IMG_H]], dtype=np.float32)
            pts_canvas = pts_img + np.array([OFFSET_X, OFFSET_Y], dtype=np.float32)
        elif key == ord('p'):
            pts = get_points_image_coords()
            print("\nCurrent points (image space, 640x480 origin at top-left):")
            for i, (x, y) in enumerate(pts, start=1):
                print(f"{i}: ({x}, {y})")
        elif key == ord('s'):
            pts = get_points_image_coords()
            print("\nSaved points (image space):")
            for i, (x, y) in enumerate(pts, start=1):
                print(f"{i}: ({x}, {y})")
            if args.save_json:
                import json
                out = {
                    "points_image_xy": pts.tolist(),  # order 1..4 connected
                    "image_size_xy": [IMG_W, IMG_H],
                    "canvas_size_xy": [CANVAS_W, CANVAS_H],
                    "offset_xy": [OFFSET_X, OFFSET_Y],
                    "note": "Points are relative to the 640x480 image; can be negative or > width/height."
                }
                with open(args.save_json, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                print(f"Also wrote JSON to: {args.save_json}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
