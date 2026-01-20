# vision/geometry.py
import numpy as np
import cv2
from config import EPS

def iris_center_pts(lm, ids):
    pts = np.array([[lm[i].x, lm[i].y] for i in ids], dtype=np.float32)
    return pts.mean(axis=0), pts

def eye_width(lm, out_idx, in_idx):
    p_out = np.array([lm[out_idx].x, lm[out_idx].y], dtype=np.float32)
    p_in  = np.array([lm[in_idx].x,  lm[in_idx].y],  dtype=np.float32)
    return float(np.linalg.norm(p_out - p_in) + EPS)

def ear(lm, top_idx, bot_idx, out_idx, in_idx):
    p_t = np.array([lm[top_idx].x, lm[top_idx].y], dtype=np.float32)
    p_b = np.array([lm[bot_idx].x, lm[bot_idx].y], dtype=np.float32)
    p_o = np.array([lm[out_idx].x, lm[out_idx].y], dtype=np.float32)
    p_i = np.array([lm[in_idx].x, lm[in_idx].y], dtype=np.float32)
    return float(np.linalg.norm(p_t - p_b) / (np.linalg.norm(p_o - p_i) + EPS))

def draw_eye_contour(canvas, lm, indices, W, H, color=(0, 255, 0)):
    points = []
    for i in indices:
        x = int(lm[i].x * W)
        y = int(lm[i].y * H)
        points.append([x, y])
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [points], True, color, 1, cv2.LINE_AA)
    for x, y in points.reshape(-1, 2):
        cv2.circle(canvas, (x, y), 1, color, -1)

def prune(deq, now, win):
    # элементы могут быть: t или (t, value)
    while deq and now - (deq[0][0] if isinstance(deq[0], tuple) else deq[0]) > win:
        deq.popleft()
