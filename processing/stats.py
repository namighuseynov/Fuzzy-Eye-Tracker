# processing/stats.py
import numpy as np
from config import EPS

def median_and_mad(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0, 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, mad

def z_star(x, med, mad):
    return float((x - med) / (mad + EPS))

def robust_to_01(z, qlow, qhigh):
    denom = (qhigh - qlow)
    if abs(denom) < 1e-8:
        return 0.5
    return float(np.clip((z - qlow) / denom, 0.0, 1.0))
