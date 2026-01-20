# ui/overlay.py
import cv2

def draw_hud(
    canvas,
    W, H,
    blink_rate, blink_e, blink_term,
    sacc_rate, sacc_e, sacc_term,
    fix_med_ms, fix_e, fix_term,
    perclos, perclos_e, perclos_term,
    cl_val, cl_term,
    st_val, st_term,
    ft_val, ft_term,
    calib_done,
    calib_left_sec,
    calib_min_samples
):
    overlay = canvas.copy()
    alpha = 0.6
    rect_h = 360
    rect_w = int(W * 0.75)
    cv2.rectangle(overlay, (0, 0), (rect_w, rect_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    x_pad = 20
    fs = 0.75
    lh = 28
    y0 = 45

    cv2.putText(canvas, "FUZZY STATE ESTIMATION (Mamdani)", (x_pad, y0),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

    cv2.putText(canvas, f"BlinkRate: {blink_rate:5.2f}/min | n={blink_e:4.2f} | {blink_term}",
                (x_pad, y0 + 1*lh), cv2.FONT_HERSHEY_SIMPLEX, fs, (100, 255, 100), 2)

    cv2.putText(canvas, f"SaccRate : {sacc_rate:5.2f}/min | n={sacc_e:4.2f} | {sacc_term}",
                (x_pad, y0 + 2*lh), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 100), 2)

    cv2.putText(canvas, f"FixDur   : {fix_med_ms:4.0f} ms | n={fix_e:4.2f} | {fix_term}",
                (x_pad, y0 + 3*lh), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 100, 100), 2)

    cv2.putText(canvas, f"PERCLOS  : {perclos*100:5.1f}% | n={perclos_e:4.2f} | {perclos_term}",
                (x_pad, y0 + 4*lh), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 100, 255), 2)

    cv2.putText(canvas, f"CL      : {cl_val:4.2f} | {cl_term}",
                (x_pad, y0 + 6*lh), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)

    cv2.putText(canvas, f"Stress  : {st_val:4.2f} | {st_term}",
                (x_pad, y0 + 7*lh), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)

    cv2.putText(canvas, f"Fatigue : {ft_val:4.2f} | {ft_term}",
                (x_pad, y0 + 8*lh), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)

    if calib_done:
        calib_text = "CALIBRATION: READY"
    else:
        calib_text = f"CALIBRATION: {calib_left_sec}s (need >= {calib_min_samples} samples)"
    cv2.putText(canvas, calib_text, (x_pad, y0 + 10*lh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.putText(canvas, "ESC to quit", (W - 200, H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)

    return canvas
