# fuzzy/membership.py
import numpy as np

def trapmf(x, a, b, c, d):
    x = float(x)
    if x <= a or x >= d: return 0.0
    if b <= x <= c: return 1.0
    if a < x < b: return (x - a) / (b - a + 1e-9)
    return (d - x) / (d - c + 1e-9)

def trimf(x, a, b, c):
    x = float(x)
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if a < x < b: return (x - a) / (b - a + 1e-9)
    return (c - x) / (c - b + 1e-9)

LMH = {
    "Low":    ("trap", (0.00, 0.00, 0.25, 0.45)),
    "Medium": ("tri",  (0.30, 0.50, 0.70)),
    "High":   ("trap", (0.55, 0.75, 1.00, 1.00)),
}

SNL = {
    "Short":  ("trap", (0.00, 0.00, 0.25, 0.45)),
    "Normal": ("tri",  (0.30, 0.50, 0.70)),
    "Long":   ("trap", (0.55, 0.75, 1.00, 1.00)),
}

def fuzzify(x01, terms_dict):
    x01 = float(np.clip(x01, 0.0, 1.0))
    out = {}
    for name, (typ, p) in terms_dict.items():
        out[name] = trapmf(x01, *p) if typ == "trap" else trimf(x01, *p)
    return out

def crisp_label(mu_dict):
    return max(mu_dict.items(), key=lambda kv: kv[1])[0]
