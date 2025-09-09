from typing import Dict

def lcr_ratio(hqla: float, net_outflows: float) -> Dict:
    """
    Liquidity Coverage Ratio = HQLA / Net Cash Outflows over 30 days.
    Returns ratio and a plain-language explanation.
    """
    if net_outflows <= 0:
        return {"ok": False, "error": "net_outflows must be > 0"}
    ratio = hqla / net_outflows
    return {
        "ok": True,
        "ratio": ratio,
        "explanation": f"LCR = {hqla} / {net_outflows} = {ratio:.4f}"
    }

def toy_var(mean: float, stdev: float, horizon_days: int, cl: float) -> Dict:
    """
    Simple Gaussian VaR (didactic, not production-ready).
    Inputs:
      - mean: daily mean PnL (same units as output)
      - stdev: daily stdev PnL
      - horizon_days: e.g., 10
      - cl: confidence level in [0,1] (e.g., 0.99)
    Returns a (positive) loss number (VaR).
    """
    if not (0.0 < cl < 1.0):
        return {"ok": False, "error": "cl must be between 0 and 1"}
    if horizon_days <= 0 or stdev < 0:
        return {"ok": False, "error": "bad inputs"}

    # z-scores for common CLs (approx)
    z_map = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263, 0.995: 2.5758}
    # nearest key (simple)
    nearest = min(z_map.keys(), key=lambda k: abs(k - cl))
    z = z_map[nearest]

    import math
    mu = mean * horizon_days
    sigma = stdev * math.sqrt(horizon_days)
    var = -(mu + z * sigma)  # loss is positive
    return {
        "ok": True,
        "var": float(max(var, 0.0)),
        "used_confidence": nearest,
        "explanation": f"VaR ≈ -({mu:.4f} + {z:.4f} * {sigma:.4f}) = {var:.4f} (loss, clipped ≥ 0)"
    }
