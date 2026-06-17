"""
ADME-T Prediction API  —  drop-in replacement for SwissADME-style workflows
============================================================================
Flask REST API serving 11 ADME-T endpoints from trained models.

Run:
    cd /path/to/adme-tdc-benchmark
    python api/app.py                    # http://127.0.0.1:5000
    python api/app.py --port 8080
    python api/app.py --host 0.0.0.0     # expose on network

Endpoints:
    GET  /                               API info
    GET  /health                         readiness check
    POST /predict   {"smiles": "..."}    predict single molecule
    POST /predict   {"smiles_list": []}  predict batch (≤1000)
    GET  /predict?smiles=<URL-encoded>   quick single-molecule query

Examples:
    curl "http://localhost:5000/predict?smiles=CC(=O)Oc1ccccc1C(=O)O"

    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'

    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"smiles_list": ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"],
              "tasks": ["solubility", "hia", "bbb"]}'
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from flask import Flask, jsonify, request

app = Flask(__name__)

# ── task registry ──────────────────────────────────────────────────────────────
# best_model: which inference function to use (matches predict_new_molecules.TASKS)
TASKS = {
    "solubility":      {"type": "regression",     "metric": "MAE↓",   "unit": "log mol/L",   "best_model": "dmpnn",  "sota": 0.761, "ours": 0.788},
    "hia":             {"type": "classification",  "metric": "AUROC↑", "unit": "probability", "best_model": "fp_gin", "sota": 0.989, "ours": 0.990},
    "caco2":           {"type": "regression",     "metric": "MAE↓",   "unit": "log cm/s",    "best_model": "fp",     "sota": 0.256, "ours": 0.276},
    "bioavailability": {"type": "classification",  "metric": "AUROC↑", "unit": "probability", "best_model": "fp_gin", "sota": 0.938, "ours": 0.743},
    "bbb":             {"type": "classification",  "metric": "AUROC↑", "unit": "probability", "best_model": "fp",     "sota": 0.916, "ours": 0.913},
    "pgp":             {"type": "classification",  "metric": "AUROC↑", "unit": "probability", "best_model": "fp_gin", "sota": 0.938, "ours": 0.933},
    "cyp1a2":          {"type": "classification",  "metric": "AUROC↑", "unit": "probability", "best_model": "fp_gin", "sota": 0.930, "ours": 0.963},
    "cyp2c19":         {"type": "classification",  "metric": "AUROC↑", "unit": "probability", "best_model": "fp_gin", "sota": 0.900, "ours": 0.930},
    "cyp2c9":          {"type": "classification",  "metric": "AUPRC↑", "unit": "probability", "best_model": "fp_gin", "sota": 0.859, "ours": 0.858},
    "cyp2d6":          {"type": "classification",  "metric": "AUPRC↑", "unit": "probability", "best_model": "fp_gin", "sota": 0.790, "ours": 0.788},
    "cyp3a4":          {"type": "classification",  "metric": "AUPRC↑", "unit": "probability", "best_model": "fp_gin", "sota": 0.916, "ours": 0.916},
}

DESCRIPTIONS = {
    "solubility":      "Aqueous solubility — AqSolDB scaffold split",
    "hia":             "Human intestinal absorption — Hou dataset",
    "caco2":           "Caco-2 membrane permeability — Wang dataset",
    "bioavailability": "Oral bioavailability (F≥20%) — Ma dataset",
    "bbb":             "Blood–brain barrier penetration — Martins dataset",
    "pgp":             "P-glycoprotein efflux substrate — Broccatelli dataset",
    "cyp1a2":          "CYP1A2 inhibition — Veith dataset",
    "cyp2c19":         "CYP2C19 inhibition — Veith dataset",
    "cyp2c9":          "CYP2C9 inhibition — Veith dataset",
    "cyp2d6":          "CYP2D6 inhibition — Veith dataset",
    "cyp3a4":          "CYP3A4 inhibition — Veith dataset",
}

# ── interpretation ─────────────────────────────────────────────────────────────

def _class_solubility(logS: float) -> str:
    if logS > -1:  return "very soluble"
    if logS > -3:  return "freely soluble"
    if logS > -5:  return "moderately soluble"
    if logS > -7:  return "slightly soluble"
    return "practically insoluble"

def _class_caco2(logP: float) -> str:
    if logP > -5.15: return "high permeability (>7.1 nm/s)"
    if logP > -6.0:  return "moderate permeability"
    return "low permeability (<1 nm/s)"

_CLASS_MAP = {
    "hia":             (0.5, "well absorbed",          "poorly absorbed"),
    "bioavailability": (0.5, "bioavailable (F≥20%)",   "low oral bioavailability"),
    "bbb":             (0.5, "CNS penetrant",           "non-CNS penetrant"),
    "pgp":             (0.5, "Pgp substrate (efflux)",  "not a Pgp substrate"),
    "cyp1a2":          (0.5, "CYP1A2 inhibitor",        "not a CYP1A2 inhibitor"),
    "cyp2c19":         (0.5, "CYP2C19 inhibitor",       "not a CYP2C19 inhibitor"),
    "cyp2c9":          (0.5, "CYP2C9 inhibitor",        "not a CYP2C9 inhibitor"),
    "cyp2d6":          (0.5, "CYP2D6 inhibitor",        "not a CYP2D6 inhibitor"),
    "cyp3a4":          (0.5, "CYP3A4 inhibitor",        "not a CYP3A4 inhibitor"),
}

def _format_prediction(task: str, raw: float | None) -> dict:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {"value": None, "status": "failed"}

    meta = TASKS[task]
    if task == "solubility":
        return {"value": round(raw, 4), "unit": meta["unit"],
                "class": _class_solubility(raw)}
    if task == "caco2":
        return {"value": round(raw, 4), "unit": meta["unit"],
                "class": _class_caco2(raw)}

    # classification
    prob = round(float(raw), 4)
    thr, pos, neg = _CLASS_MAP[task]
    return {"probability": prob, "class": pos if prob >= thr else neg}

# ── model warm-up ──────────────────────────────────────────────────────────────

_warmed_up = False

def _warmup():
    global _warmed_up
    if _warmed_up:
        return
    from features import smiles_to_features
    smiles_to_features("c1ccccc1")     # validates RDKit stack once
    _warmed_up = True

# ── core prediction ────────────────────────────────────────────────────────────

def _run_predictions(smiles_list: list[str], tasks: list[str]) -> dict[str, list]:
    """Return {task: [value_or_None, ...]} for every molecule."""
    from predict_new_molecules import (
        predict_fp, predict_fp_gin, predict_dmpnn,
        TASKS as _PM_TASKS,
    )

    fp_tasks    = [t for t in tasks if _PM_TASKS[t][2] == "fp"]
    fpgin_tasks = [t for t in tasks if _PM_TASKS[t][2] == "fp_gin"]
    dmpnn_tasks = [t for t in tasks if _PM_TASKS[t][2] == "dmpnn"]

    out: dict[str, list] = {}

    for fn, task_list in [
        (lambda tl: predict_dmpnn(smiles_list, tl),  dmpnn_tasks),
        (lambda tl: predict_fp(smiles_list, tl),     fp_tasks),
        (lambda tl: predict_fp_gin(smiles_list, tl), fpgin_tasks),
    ]:
        if not task_list:
            continue
        df = fn(task_list)
        for t in task_list:
            out[t] = df[t].tolist() if t in df.columns else [None] * len(smiles_list)

    return out

def _predict_and_format(smiles_list: list[str], tasks: list[str]) -> list[dict]:
    from rdkit import Chem

    raw = _run_predictions(smiles_list, tasks)

    results = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            results.append({
                "smiles": smi,
                "canonical_smiles": None,
                "status": "invalid_smiles",
                "predictions": {},
            })
            continue

        preds = {t: _format_prediction(t, raw.get(t, [None] * len(smiles_list))[i])
                 for t in tasks}
        results.append({
            "smiles": smi,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "status": "ok",
            "predictions": preds,
        })
    return results

# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "ADME-T Prediction API",
        "version": "1.0",
        "description": (
            "11-endpoint ADME-T prediction from trained CatBoost+GIN "
            "and D-MPNN models. Verified SOTA or near-SOTA on TDC ADME "
            "scaffold splits (Koleiev 2026 audit). "
            "Analogous to SwissADME but backed by trained ML models."
        ),
        "usage": {
            "single_GET":  "GET /predict?smiles=<URL-encoded SMILES>",
            "single_POST": "POST /predict   body: {\"smiles\": \"...\"}",
            "batch_POST":  "POST /predict   body: {\"smiles_list\": [...], \"tasks\": [...]}",
        },
        "tasks": {
            k: {
                "description": DESCRIPTIONS[k],
                "metric": v["metric"],
                "unit": v["unit"],
                "our_score": v["ours"],
                "sota": v["sota"],
            }
            for k, v in TASKS.items()
        },
    })


@app.route("/health", methods=["GET"])
def health():
    try:
        _warmup()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 503


@app.route("/predict", methods=["GET", "POST"])
def predict():
    t0 = time.time()

    try:
        _warmup()
    except Exception as e:
        return jsonify({"error": f"Initialization error: {e}"}), 503

    # ── parse input ───────────────────────────────────────────────────────────
    if request.method == "GET":
        smi = request.args.get("smiles", "").strip()
        if not smi:
            return jsonify({"error": "Provide ?smiles=<SMILES>"}), 400
        smiles_list = [smi]
        tasks = list(TASKS.keys())

    else:
        body = request.get_json(silent=True) or {}
        if "smiles_list" in body:
            smiles_list = [str(s).strip() for s in body["smiles_list"] if str(s).strip()]
        elif "smiles" in body:
            smiles_list = [str(body["smiles"]).strip()]
        else:
            return jsonify({"error": "'smiles' or 'smiles_list' required"}), 400

        if len(smiles_list) > 1000:
            return jsonify({"error": "Batch limited to 1000 molecules per request"}), 400

        raw_tasks = body.get("tasks", list(TASKS.keys()))
        tasks = [t for t in raw_tasks if t in TASKS]
        if not tasks:
            return jsonify({"error": f"No valid tasks. Available: {list(TASKS.keys())}"}), 400

    if not smiles_list:
        return jsonify({"error": "Empty input"}), 400

    # ── predict ───────────────────────────────────────────────────────────────
    try:
        results = _predict_and_format(smiles_list, tasks)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

    n_ok = sum(1 for r in results if r["status"] == "ok")
    elapsed = round(time.time() - t0, 3)

    resp = {
        "n_molecules": len(smiles_list),
        "n_predicted": n_ok,
        "tasks": tasks,
        "elapsed_s": elapsed,
        "results": results,
    }
    if len(smiles_list) == 1:
        resp["result"] = results[0]

    return jsonify(resp)

# ── entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="ADME-T Prediction API")
    p.add_argument("--host",  default="127.0.0.1")
    p.add_argument("--port",  type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    print(f"\nADME-T API  →  http://{args.host}:{args.port}")
    print("  GET  /                API info")
    print("  GET  /health          readiness")
    print("  GET  /predict?smiles= single molecule")
    print("  POST /predict         batch prediction\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
