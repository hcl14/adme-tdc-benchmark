"""
Collect every results/<task>/result*.json into a single best-per-task table.

For each task picks the best model by the metric direction (↑ AUROC/AUPRC,
↓ MAE) and emits both a console table and results/final_summary.{json,md}.
"""
from __future__ import annotations
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"

TASKS = {
    "solubility"      : ("Solubility_AqSolDB", "regression", "mae",   0.761),
    "hia"             : ("HIA_Hou",            "binary",     "auroc", 0.989),
    "caco2"           : ("Caco2_Wang",         "regression", "mae",   0.256),
    "bioavailability" : ("Bioavailability_Ma", "binary",     "auroc", 0.938),
    "bbb"             : ("BBB_Martins",        "binary",     "auroc", 0.916),
    "pgp"             : ("Pgp_Broccatelli",    "binary",     "auroc", 0.938),
    "cyp1a2"          : ("CYP1A2_Veith",       "binary",     "auroc", 0.930),
    "cyp2c19"         : ("CYP2C19_Veith",      "binary",     "auroc", 0.900),
    "cyp2c9"          : ("CYP2C9_Veith",       "binary",     "auprc", 0.859),
    "cyp2d6"          : ("CYP2D6_Veith",       "binary",     "auprc", 0.790),
    "cyp3a4"          : ("CYP3A4_Veith",       "binary",     "auprc", 0.916),
}


def better(metric, a, b):
    """Is score a better than b for this metric?"""
    if a is None:
        return False
    if b is None:
        return True
    return (a > b) if metric != "mae" else (a < b)


def load_all(task):
    out = []
    d = R / task
    if not d.exists():
        return out
    for f in sorted(d.glob("result*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        mean = j.get("mean", j.get("mae"))
        if mean is None:
            continue
        out.append({"file": f.name, "model": j.get("model", f.stem),
                    "mean": float(mean), "std": float(j.get("std", 0) or 0),
                    "n": len(j.get("seeds", []))})
    return out


def main():
    rows = []
    for tkey, (tdc, ttype, metric, sota) in TASKS.items():
        allm = load_all(tkey)
        best = None
        for m in allm:
            if better(metric, m["mean"], best["mean"] if best else None):
                best = m
        if best is None:
            rows.append({"task": tkey, "metric": metric, "sota": sota,
                         "best": None, "all": allm})
            continue
        delta = (sota - best["mean"]) if metric == "mae" else (best["mean"] - sota)
        rows.append({"task": tkey, "metric": metric, "sota": sota,
                     "best": best, "delta": delta, "all": allm})

    # console
    print("=" * 92)
    print(f"{'task':<14}{'metric':<7}{'best model':<16}{'mean':>9}{'±std':>9}"
          f"{'sota':>8}{'Δ':>9}  status")
    print("-" * 92)
    n_sota = 0
    for r in rows:
        if r["best"] is None:
            print(f"  {r['task']:<12}{r['metric']:<7} (no result)")
            continue
        b = r["best"]
        at = "✅" if (r["delta"] >= -0.005) else ("≈" if r["delta"] >= -0.02 else "✗")
        if r["delta"] >= -0.005:
            n_sota += 1
        print(f"  {r['task']:<12}{r['metric']:<7}{b['model']:<16}{b['mean']:>9.4f}"
              f"{b['std']:>9.4f}{r['sota']:>8.3f}{r['delta']:>+9.4f}  {at} "
              f"({b['n']} seeds)")
    print("=" * 92)
    print(f"At/within SOTA (|Δ|<=0.005): {n_sota}/{len(rows)}")

    # json
    with open(R / "final_summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    # markdown
    lines = ["# ADME-T Final Results (best model per task)\n",
             "| Task | Metric | Model | Ours (mean±std) | SOTA | Δ | Status |",
             "|------|--------|-------|-----------------|------|---|--------|"]
    for r in rows:
        if r["best"] is None:
            lines.append(f"| {r['task']} | {r['metric']} | — | — | {r['sota']} | — | pending |")
            continue
        b = r["best"]
        at = "✅ at SOTA" if r["delta"] >= -0.005 else ("≈ near" if r["delta"] >= -0.02 else "✗ gap")
        dirn = "↓" if r["metric"] == "mae" else "↑"
        lines.append(f"| {r['task']} | {r['metric'].upper()}{dirn} | {b['model']} | "
                     f"{b['mean']:.4f}±{b['std']:.4f} | {r['sota']:.3f} | "
                     f"{r['delta']:+.4f} | {at} |")
    lines.append("")
    with open(R / "final_summary.md", "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved {R/'final_summary.json'} and {R/'final_summary.md'}")


if __name__ == "__main__":
    main()
