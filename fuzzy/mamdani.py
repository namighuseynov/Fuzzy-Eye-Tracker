# fuzzy/mamdani.py
import numpy as np
from .membership import trapmf, trimf, fuzzify, crisp_label, LMH

def mf_eval(x, mf_spec):
    typ, p = mf_spec
    return trapmf(x, *p) if typ == "trap" else trimf(x, *p)

def aggregate_output_curve(term_strengths, x_grid, out_terms=LMH):
    mu = np.zeros_like(x_grid, dtype=np.float32)
    for term, strength in term_strengths.items():
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0:
            continue
        mf_spec = out_terms[term]
        mf_vals = np.array([mf_eval(x, mf_spec) for x in x_grid], dtype=np.float32)
        mu_term = np.minimum(mf_vals, strength)
        mu = np.maximum(mu, mu_term)
    return mu

def defuzz_centroid(x_grid, mu_curve):
    num = float(np.sum(x_grid * mu_curve))
    den = float(np.sum(mu_curve) + 1e-12)
    if den < 1e-8:
        return 0.5
    return num / den

def mamdani_infer(mu_fix, mu_sacc, mu_blink, mu_perc):
    def AND(*vals): return float(min(vals))
    def OR(*vals):  return float(max(vals))

    cl_terms = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
    cl_terms["High"] = max(
        cl_terms["High"],
        AND(mu_fix["Long"], mu_sacc["High"]),
        AND(mu_fix["Long"], mu_sacc["Medium"]),
        AND(mu_sacc["High"], mu_perc["Low"]),
        AND(mu_fix["Long"], mu_blink["Medium"]),
    )
    cl_terms["Medium"] = max(
        cl_terms["Medium"],
        AND(mu_fix["Normal"], mu_sacc["Medium"]),
        AND(mu_fix["Normal"], OR(mu_sacc["Low"], mu_sacc["Medium"])),
        AND(mu_fix["Long"], mu_sacc["Low"]),
    )
    cl_terms["Low"] = max(
        cl_terms["Low"],
        AND(mu_fix["Short"], mu_sacc["Low"]),
        AND(mu_fix["Short"], mu_sacc["Medium"]),
        mu_perc["High"],
    )

    st_terms = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
    st_terms["High"] = max(
        st_terms["High"],
        AND(mu_sacc["High"], mu_fix["Long"]),
        AND(mu_sacc["High"], mu_blink["High"]),
        AND(mu_fix["Long"], mu_blink["High"]),
    )
    st_terms["Medium"] = max(
        st_terms["Medium"],
        AND(mu_sacc["Medium"], mu_fix["Normal"]),
        AND(mu_sacc["Medium"], mu_blink["Medium"]),
        AND(mu_fix["Normal"], mu_blink["High"]),
    )
    st_terms["Low"] = max(
        st_terms["Low"],
        AND(mu_sacc["Low"], OR(mu_fix["Short"], mu_fix["Normal"])),
        AND(mu_blink["Low"], mu_sacc["Low"]),
        mu_perc["High"],
    )

    ft_terms = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
    ft_terms["High"] = max(
        ft_terms["High"],
        mu_perc["High"],
        AND(mu_perc["Medium"], mu_blink["High"]),
        AND(mu_sacc["Low"], mu_blink["High"], mu_fix["Long"]),
    )
    ft_terms["Medium"] = max(
        ft_terms["Medium"],
        mu_perc["Medium"],
        AND(mu_blink["Medium"], mu_sacc["Low"]),
        AND(mu_blink["High"], mu_sacc["Medium"]),
    )
    ft_terms["Low"] = max(
        ft_terms["Low"],
        AND(mu_perc["Low"], mu_sacc["High"]),
        AND(mu_perc["Low"], mu_blink["Low"]),
    )

    x_grid = np.linspace(0.0, 1.0, 101, dtype=np.float32)

    cl_curve = aggregate_output_curve(cl_terms, x_grid, LMH)
    st_curve = aggregate_output_curve(st_terms, x_grid, LMH)
    ft_curve = aggregate_output_curve(ft_terms, x_grid, LMH)

    cl_val = defuzz_centroid(x_grid, cl_curve)
    st_val = defuzz_centroid(x_grid, st_curve)
    ft_val = defuzz_centroid(x_grid, ft_curve)

    cl_label = crisp_label(fuzzify(cl_val, LMH))
    st_label = crisp_label(fuzzify(st_val, LMH))
    ft_label = crisp_label(fuzzify(ft_val, LMH))

    return (cl_terms, cl_val, cl_label), (st_terms, st_val, st_label), (ft_terms, ft_val, ft_label)
