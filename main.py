# main.py
import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime


from config import (
    CAM_INDEX, CAP_FPS,
    WINDOW_SEC, UPDATE_HZ,
    CALIB_SEC, CALIB_MIN_SAMPLES, Q_LOW, Q_HIGH,
    EWMA_ALPHA, SMOOTH_N,
    BLINK_THR, BLINK_MIN_FRAMES,
    SAC_VEL_THR, SAC_MIN_FRAMES,
    FIX_VEL_THR, FIX_MIN_MS,
    EPS,
)

from vision.landmarks import (
    L_OUT, L_IN, R_OUT, R_IN,
    L_IRIS, R_IRIS,
    L_TOP, L_BOT, R_TOP, R_BOT,
    LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR
)

from vision.geometry import (
    iris_center_pts, eye_width, ear,
    draw_eye_contour
)

from vision.events import EventDetector

from processing.stats import median_and_mad, z_star, robust_to_01
from processing.smoothing import ewma
from processing.perclos import perclos_time_weighted

from fuzzy.membership import fuzzify, crisp_label, LMH, SNL
from fuzzy.mamdani import mamdani_infer

from storage.logger import CSVLogger
from storage.udp_sender import UdpSender
from ui.overlay import draw_hud


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"eye_metrics_{stamp}.csv"

    # окно
    W, H = 1920, 1080
    cv2.namedWindow("EYE METRICS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("EYE METRICS", W, H)

    # камера
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    assert cap.isOpened(), "Камера не открылась"
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    # detector
    det = EventDetector(
        window_sec=WINDOW_SEC,
        blink_thr=BLINK_THR,
        blink_min_frames=BLINK_MIN_FRAMES,
        sac_vel_thr=SAC_VEL_THR,
        sac_min_frames=SAC_MIN_FRAMES,
        fix_vel_thr=FIX_VEL_THR,
        fix_min_ms=FIX_MIN_MS,
        smooth_n=SMOOTH_N,
    )

    # калибровка
    calib_start = time.time()
    calib_done = False
    base_raw = {"blink_rate": [], "sacc_rate": [], "fix_med_ms": [], "perclos": []}
    base_stats = {}
    base_q = {}
    ew_state = {"blink": None, "sacc": None, "fix": None, "perclos": None}

    # лог
    logger = CSVLogger(out_csv)
    logger.write_header([
        "t",
        "blink_rate_per_min", "saccade_rate_per_min",
        "fix_dur_median_ms", "fix_dur_mean_ms",
        "PERCLOS",
        "blink_n_ewma", "sacc_n_ewma", "fix_n_ewma", "perclos_n_ewma",
        "FixDur_term", "SaccRate_term", "BlinkRate_term", "PERCLOS_term",
        "CL_value", "CL_term",
        "Stress_value", "Stress_term",
        "Fatigue_value", "Fatigue_term",
        "calib_ready"
    ])

    udp = UdpSender("127.0.0.1", 5005)

    # состояние для gaze smoothing (как у тебя: по двум глазам)
    bufL, bufR = deque(maxlen=SMOOTH_N), deque(maxlen=SMOOTH_N)
    prev_g, prev_t = None, None

    # текущие метрики (HUD)
    blink_rate = sacc_rate = 0.0
    fix_med_ms = fix_mean_ms = 0.0
    perclos = 0.0

    blink_e = sacc_e = fix_e = perclos_e = 0.5
    fix_term = sacc_term = blink_term = perclos_term = "NA"
    cl_val = st_val = ft_val = 0.5
    cl_term = st_term = ft_term = "Medium"

    # таймер обновления
    next_emit_t = time.time()
    emit_dt = 1.0 / max(1, UPDATE_HZ)

    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            t_now = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)

            canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            canvas = cv2.resize(canvas, (W, H))
            gray_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            canvas = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            face_ok = bool(res.multi_face_landmarks)

            if face_ok:
                lm = res.multi_face_landmarks[0].landmark

                (cL, _) = iris_center_pts(lm, L_IRIS)
                (cR, _) = iris_center_pts(lm, R_IRIS)
                wL, wR = eye_width(lm, L_OUT, L_IN), eye_width(lm, R_OUT, R_IN)

                L_mid = 0.5 * (np.array([lm[L_OUT].x, lm[L_OUT].y], dtype=np.float32) +
                               np.array([lm[L_IN].x,  lm[L_IN].y],  dtype=np.float32))
                R_mid = 0.5 * (np.array([lm[R_OUT].x, lm[R_OUT].y], dtype=np.float32) +
                               np.array([lm[R_IN].x,  lm[R_IN].y],  dtype=np.float32))

                gL = (cL - L_mid) / (wL + EPS)
                gR = (cR - R_mid) / (wR + EPS)

                bufL.append(gL)
                bufR.append(gR)
                g = (np.mean(bufL, axis=0) + np.mean(bufR, axis=0)) / 2.0

                # EAR + closed
                earL = ear(lm, L_TOP, L_BOT, L_OUT, L_IN)
                earR = ear(lm, R_TOP, R_BOT, R_OUT, R_IN)
                ear_avg = 0.5 * (earL + earR)

                # update events
                det.update(t_now, g, ear_avg)
                det.prune(t_now)

                # PERCLOS по времени
                perclos = perclos_time_weighted(det.closed_flags, t_now, WINDOW_SEC)

                # debug overlay
                draw_eye_contour(canvas, lm, LEFT_EYE_CONTOUR, W, H, (0, 255, 0))
                draw_eye_contour(canvas, lm, RIGHT_EYE_CONTOUR, W, H, (0, 255, 0))
                cv2.circle(canvas, (int(cL[0] * W), int(cL[1] * H)), 3, (0, 165, 255), -1)
                cv2.circle(canvas, (int(cR[0] * W), int(cR[1] * H)), 3, (0, 165, 255), -1)

            else:
                cv2.putText(canvas, "Face not detected", (15, H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            # ---------- UPDATE 1 Hz ----------
            if t_now >= next_emit_t:
                if face_ok:
                    blink_rate, sacc_rate, fix_med_ms, fix_mean_ms = det.compute_metrics()
                else:
                    blink_rate = sacc_rate = 0.0
                    fix_med_ms = fix_mean_ms = 0.0
                    perclos = 0.0

                # collect baseline
                if (not calib_done) and face_ok:
                    base_raw["blink_rate"].append(blink_rate)
                    base_raw["sacc_rate"].append(sacc_rate)
                    base_raw["fix_med_ms"].append(fix_med_ms)
                    base_raw["perclos"].append(perclos)

                    enough_time = (t_now - calib_start) >= CALIB_SEC
                    enough_samples = len(base_raw["blink_rate"]) >= CALIB_MIN_SAMPLES

                    if enough_time and enough_samples:
                        for k in base_raw:
                            med, mad = median_and_mad(base_raw[k])
                            base_stats[k] = (med, mad)

                        for k in base_raw:
                            med, mad = base_stats[k]
                            z_samples = [z_star(v, med, mad) for v in base_raw[k]]
                            ql = float(np.quantile(z_samples, Q_LOW))
                            qh = float(np.quantile(z_samples, Q_HIGH))
                            base_q[k] = (ql, qh)

                        calib_done = True
                        ew_state = {"blink": None, "sacc": None, "fix": None, "perclos": None}

                # normalize + EWMA
                if calib_done:
                    med, mad = base_stats["blink_rate"]; ql, qh = base_q["blink_rate"]
                    blink_n = robust_to_01(z_star(blink_rate, med, mad), ql, qh)
                    ew_state["blink"] = ewma(ew_state["blink"], blink_n, EWMA_ALPHA)
                    blink_e = ew_state["blink"]

                    med, mad = base_stats["sacc_rate"]; ql, qh = base_q["sacc_rate"]
                    sacc_n = robust_to_01(z_star(sacc_rate, med, mad), ql, qh)
                    ew_state["sacc"] = ewma(ew_state["sacc"], sacc_n, EWMA_ALPHA)
                    sacc_e = ew_state["sacc"]

                    med, mad = base_stats["fix_med_ms"]; ql, qh = base_q["fix_med_ms"]
                    fix_n = robust_to_01(z_star(fix_med_ms, med, mad), ql, qh)
                    ew_state["fix"] = ewma(ew_state["fix"], fix_n, EWMA_ALPHA)
                    fix_e = ew_state["fix"]

                    med, mad = base_stats["perclos"]; ql, qh = base_q["perclos"]
                    perclos_n = robust_to_01(z_star(perclos, med, mad), ql, qh)
                    ew_state["perclos"] = ewma(ew_state["perclos"], perclos_n, EWMA_ALPHA)
                    perclos_e = ew_state["perclos"]
                else:
                    blink_e = sacc_e = fix_e = perclos_e = 0.5

                # fuzzify inputs
                mu_fix = fuzzify(fix_e, SNL)
                mu_sacc = fuzzify(sacc_e, LMH)
                mu_blink = fuzzify(blink_e, LMH)
                mu_perc = fuzzify(perclos_e, LMH)

                fix_term = crisp_label(mu_fix)
                sacc_term = crisp_label(mu_sacc)
                blink_term = crisp_label(mu_blink)
                perclos_term = crisp_label(mu_perc)

                # infer outputs
                (cl_terms, cl_val, cl_term), (st_terms, st_val, st_term), (ft_terms, ft_val, ft_term) = mamdani_infer(
                    mu_fix, mu_sacc, mu_blink, mu_perc
                )

                # write CSV
                logger.write_row([
                    t_now,
                    blink_rate, sacc_rate,
                    fix_med_ms, fix_mean_ms,
                    perclos,
                    blink_e, sacc_e, fix_e, perclos_e,
                    fix_term, sacc_term, blink_term, perclos_term,
                    cl_val, cl_term,
                    st_val, st_term,
                    ft_val, ft_term,
                    1 if calib_done else 0
                ])

                udp.send({
                    "t": t_now,
                    "blink_rate": blink_rate,
                    "sacc_rate": sacc_rate,
                    "fix_med_ms": fix_med_ms,
                    "perclos": perclos,

                    "blink_n": blink_e,
                    "sacc_n": sacc_e,
                    "fix_n": fix_e,
                    "perclos_n": perclos_e,

                    "fix_term": fix_term,
                    "sacc_term": sacc_term,
                    "blink_term": blink_term,
                    "perclos_term": perclos_term,

                    "cl": cl_val,
                    "cl_term": cl_term,
                    "stress": st_val,
                    "stress_term": st_term,
                    "fatigue": ft_val,
                    "fatigue_term": ft_term,

                    "calib_ready": int(calib_done)
                })

                next_emit_t = t_now + emit_dt

            # ---------- HUD ----------
            calib_left = int(max(0, CALIB_SEC - (time.time() - calib_start)))
            canvas = draw_hud(
                canvas, W, H,
                blink_rate, blink_e, blink_term,
                sacc_rate, sacc_e, sacc_term,
                fix_med_ms, fix_e, fix_term,
                perclos, perclos_e, perclos_term,
                cl_val, cl_term,
                st_val, st_term,
                ft_val, ft_term,
                calib_done,
                calib_left,
                CALIB_MIN_SAMPLES
            )

            cv2.imshow("EYE METRICS", canvas)
            if cv2.waitKey(1) == 27:
                break

    logger.close()
    udp.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
