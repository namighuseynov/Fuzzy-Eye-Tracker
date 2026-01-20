# processing/perclos.py
import numpy as np

def perclos_time_weighted(closed_flags, now, win_sec):
    if len(closed_flags) < 2:
        return 0.0

    t0 = now - win_sec
    samples = []
    prev = None

    for t, v in closed_flags:
        if t < t0:
            prev = (t, v)
            continue
        if prev is not None:
            samples.append((t0, prev[1]))
            prev = None
        samples.append((t, v))

    if not samples:
        return float(closed_flags[-1][1])

    if samples[-1][0] < now:
        samples.append((now, samples[-1][1]))

    closed_time = 0.0
    for (t1, v1), (t2, _) in zip(samples, samples[1:]):
        dt = max(0.0, t2 - t1)
        if v1 == 1:
            closed_time += dt

    return float(np.clip(closed_time / win_sec, 0.0, 1.0))
