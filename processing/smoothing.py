# processing/smoothing.py
def ewma(prev, x, alpha):
    if prev is None:
        return float(x)
    return float(alpha * x + (1.0 - alpha) * prev)
