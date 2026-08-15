"""E11: refit the anchor budget curve with a free exponent.

The original E3 code pinned the exponent at -1/2 and fitted only the constant,
so the reported R^2 could not test the rate; the fit was circular.  Here the
exponent is estimated.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations
import glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e11_anchor_powerlaw_free.json")
NS = [5, 10, 20, 50, 100, 200]

def main() -> None:
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "e3_seed*_results.json"))):
        d = json.load(open(f))["M3_anchor_budget"]
        rows.append([d[str(k)]["mean_rmse"] for k in NS])
    R = np.array(rows)
    n = np.array(NS, dtype=float)
    y = R.mean(axis=0)
    ln, ly = np.log(n), np.log(y)
    A = np.vstack([ln, np.ones_like(ln)]).T
    coef, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    slope, inter = coef
    resid = ly - A @ coef
    ss_res, ss_tot = float((resid ** 2).sum()), float(((ly - ly.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    se = np.sqrt(ss_res / (len(n) - 2) / ((ln - ln.mean()) ** 2).sum())
    t975 = 2.776  # t(0.975, df=4)
    lo, hi = slope - t975 * se, slope + t975 * se
    c5 = np.exp((ly + 0.5 * ln).mean())
    r25 = 1 - float(((ly - (-0.5 * ln + np.log(c5))) ** 2).sum()) / ss_tot
    res = {
        "env": {"script": "e11_anchor_powerlaw_free.py", "n_seeds": len(rows),
                "e3_source_files": [os.path.basename(f) for f in sorted(glob.glob(os.path.join(ROOT, "results", "e3_seed*_results.json")))],
                "deterministic": True,
                "gpu_used": False, "api_calls": 0, "cost_usd": 0.0},
        "n": NS, "rmse_mean": y.tolist(),
        "free_slope": float(slope), "free_slope_ci95": [float(lo), float(hi)],
        "free_r2": float(r2), "free_c": float(np.exp(inter)),
        "fixed_half_c": float(c5), "fixed_half_r2": float(r25),
        "minus_half_in_ci": bool(lo <= -0.5 <= hi),
        "note": ("The earlier report pinned the exponent at -0.5 and fitted only the "
                 "constant, which cannot test the rate. Theorem T2 is a worst-case "
                 "lower bound and is not verified by an estimator's observed rate."),
    }
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"free slope = {slope:.4f} CI95 = [{lo:.4f}, {hi:.4f}] R2 = {r2:.4f}")
    print(f"-0.5 in CI: {res['minus_half_in_ci']}")

if __name__ == "__main__":
    main()
