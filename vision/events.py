# vision/events.py
import numpy as np
from collections import deque
from config import EPS

class EventDetector:
    """
    Держит состояния для:
    - blink_times (по frames closed)
    - sac_times (по velocity threshold)
    - fix_durations (по velocity < threshold и min duration)
    - closed_flags (для PERCLOS)

    Использование:
      det = EventDetector(window_sec=10, smooth_n=5, ...)
      det.update(t_now, g, ear_avg)
      det.prune(t_now)
      metrics = det.compute_metrics(window_sec)
    """

    def __init__(
        self,
        window_sec,
        blink_thr,
        blink_min_frames,
        sac_vel_thr,
        sac_min_frames,
        fix_vel_thr,
        fix_min_ms,
        smooth_n=5,
    ):
        self.window_sec = window_sec

        self.blink_thr = blink_thr
        self.blink_min_frames = blink_min_frames

        self.sac_vel_thr = sac_vel_thr
        self.sac_min_frames = sac_min_frames

        self.fix_vel_thr = fix_vel_thr
        self.fix_min_ms = fix_min_ms

        self.bufL = deque(maxlen=smooth_n)  # если захотишь раздельно по глазам
        self.bufR = deque(maxlen=smooth_n)

        self.prev_g = None
        self.prev_t = None

        self.blink_counter = 0
        self.blink_active = False
        self.blink_times = deque()

        self.sac_is_on = False
        self.sac_on_frames = 0
        self.sac_times = deque()

        self.fix_is_on = False
        self.fix_start_t = 0.0
        self.fix_durations = deque()  # (t, dur_sec)

        self.closed_flags = deque()   # (t, 0/1)

    def update(self, t_now, g, ear_avg):
        # closure flag
        is_closed = ear_avg < self.blink_thr
        self.closed_flags.append((t_now, 1 if is_closed else 0))

        # blink event
        if is_closed:
            self.blink_counter += 1
            if (not self.blink_active) and self.blink_counter >= self.blink_min_frames:
                self.blink_active = True
                self.blink_times.append(t_now)
        else:
            self.blink_counter = 0
            self.blink_active = False

        # saccade + fixation
        if self.prev_g is not None:
            dt = max(EPS, t_now - self.prev_t)
            vel = float(np.linalg.norm(g - self.prev_g) / dt)

            # saccade
            if vel > self.sac_vel_thr:
                self.sac_on_frames += 1
                if (not self.sac_is_on) and self.sac_on_frames >= self.sac_min_frames:
                    self.sac_is_on = True
                    self.sac_times.append(t_now)
            else:
                self.sac_is_on = False
                self.sac_on_frames = 0

            # fixation
            if vel < self.fix_vel_thr:
                if not self.fix_is_on:
                    self.fix_is_on = True
                    self.fix_start_t = self.prev_t
            else:
                if self.fix_is_on:
                    dur = t_now - self.fix_start_t
                    if dur * 1000.0 >= self.fix_min_ms:
                        self.fix_durations.append((t_now, dur))
                    self.fix_is_on = False

        self.prev_g, self.prev_t = g, t_now

    def prune(self, t_now):
        win = self.window_sec

        def _prune(dq):
            while dq and t_now - (dq[0][0] if isinstance(dq[0], tuple) else dq[0]) > win:
                dq.popleft()

        _prune(self.blink_times)
        _prune(self.sac_times)
        _prune(self.fix_durations)
        _prune(self.closed_flags)

    def compute_metrics(self):
        # blink/sacc in per min
        blink_rate = len(self.blink_times) / self.window_sec * 60.0
        sacc_rate = len(self.sac_times) / self.window_sec * 60.0

        durs = [d for _, d in self.fix_durations]
        if durs:
            fix_mean_ms = float(np.mean(durs) * 1000.0)
            fix_med_ms = float(np.median(durs) * 1000.0)
        else:
            fix_mean_ms = 0.0
            fix_med_ms = 0.0

        return blink_rate, sacc_rate, fix_med_ms, fix_mean_ms
