# ADME-T API Contract

REST API for predicting 11 ADME-T endpoints from SMILES strings.
Trained on TDC benchmark datasets; verified SOTA or near-SOTA on all tasks (Koleiev 2026 audit).

## Starting the server

```bash
cd /path/to/adme-tdc-benchmark
python api/app.py                     # http://127.0.0.1:5000 (default)
python api/app.py --port 8080
python api/app.py --host 0.0.0.0      # expose on local network
python api/app.py --debug             # Flask debug mode
```

**First request is slow** (~5–15 s) while models load into memory. Subsequent requests are fast (<1 s for FP-only, ~2–5 s for FP+GIN batch).

---

## Endpoints

### `GET /`

Returns API metadata, version, and the full task catalogue.

**Response**

```json
{
  "name": "ADME-T Prediction API",
  "version": "1.0",
  "description": "...",
  "usage": {
    "single_GET":  "GET /predict?smiles=<URL-encoded SMILES>",
    "single_POST": "POST /predict   body: {\"smiles\": \"...\"}",
    "batch_POST":  "POST /predict   body: {\"smiles_list\": [...], \"tasks\": [...]}"
  },
  "tasks": {
    "solubility": {
      "description": "Aqueous solubility — AqSolDB scaffold split",
      "metric": "MAE↓",
      "unit": "log mol/L",
      "our_score": 0.788,
      "sota": 0.761
    },
    "...": "..."
  }
}
```

---

### `GET /health`

Readiness check — verifies RDKit is importable and the feature pipeline works.

**Response `200`**
```json
{"status": "ok"}
```

**Response `503`** (if RDKit or model files are broken)
```json
{"status": "error", "detail": "<error message>"}
```

---

### `GET /predict`

Predict all 11 endpoints for a single molecule via query parameter.

**Query parameters**

| Parameter | Type   | Required | Description                     |
|-----------|--------|----------|---------------------------------|
| `smiles`  | string | yes      | SMILES string (URL-encoded)      |

**Example**

```bash
curl "http://localhost:5000/predict?smiles=CC(=O)Oc1ccccc1C(=O)O"
```

Predicts all 11 tasks. Equivalent to `POST /predict` with `{"smiles": "..."}`.

---

### `POST /predict`

Predict one or many molecules. Optionally filter to a subset of tasks.

**Request body** (`Content-Type: application/json`)

| Field          | Type            | Required | Default        | Description                                  |
|----------------|-----------------|----------|----------------|----------------------------------------------|
| `smiles`       | string          | one of   | —              | Single SMILES string                         |
| `smiles_list`  | array\<string\> | one of   | —              | Batch of SMILES strings (max 1000)           |
| `tasks`        | array\<string\> | no       | all 11 tasks   | Subset of task keys to predict               |

Exactly one of `smiles` or `smiles_list` is required.

**Examples**

```json
{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
```

```json
{
  "smiles_list": ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCO"],
  "tasks": ["solubility", "hia", "bbb", "cyp3a4"]
}
```

---

## Response Schema

### Top-level

| Field          | Type    | Description                                               |
|----------------|---------|-----------------------------------------------------------|
| `n_molecules`  | int     | Number of SMILES received                                 |
| `n_predicted`  | int     | Number successfully predicted (valid SMILES only)         |
| `tasks`        | array   | Task keys that were predicted                             |
| `elapsed_s`    | float   | Wall-clock time for the whole request (seconds)           |
| `results`      | array   | One entry per input molecule (see below)                  |
| `result`       | object  | Shorthand alias — only present when input was 1 molecule  |

### Per-molecule object (`results[i]`)

| Field              | Type          | Description                                              |
|--------------------|---------------|----------------------------------------------------------|
| `smiles`           | string        | Input SMILES as received                                 |
| `canonical_smiles` | string\|null  | RDKit canonical form; `null` if SMILES is invalid        |
| `status`           | string        | `"ok"` or `"invalid_smiles"`                             |
| `predictions`      | object        | Map of task key → prediction object (empty if invalid)   |

### Prediction object — regression tasks (`solubility`, `caco2`)

| Field   | Type   | Description                               |
|---------|--------|-------------------------------------------|
| `value` | float  | Predicted value in the task's native unit |
| `unit`  | string | Unit string (see task table below)        |
| `class` | string | Human-readable interpretation             |

### Prediction object — classification tasks (all others)

| Field         | Type   | Description                                    |
|---------------|--------|------------------------------------------------|
| `probability` | float  | Predicted probability of the positive class    |
| `class`       | string | Human-readable label at threshold 0.5          |

### Failed prediction (invalid SMILES or missing model file)

```json
{"value": null, "status": "failed"}
```

---

## Task Reference

### Regression tasks

| Task key      | Output unit | Range (typical) | Interpretation thresholds              | Model  |
|---------------|-------------|-----------------|----------------------------------------|--------|
| `solubility`  | `log mol/L` | −7 to 0         | >−1: very soluble; −1 to −3: freely; −3 to −5: moderate; −5 to −7: slight; <−7: insoluble | D-MPNN |
| `caco2`       | `log cm/s`  | −7 to −4        | >−5.15: high (>7.1 nm/s); −5.15 to −6: moderate; <−6: low (<1 nm/s) | FP     |

### Classification tasks

| Task key        | Positive label             | Negative label                | Model  | TDC dataset          |
|-----------------|---------------------------|-------------------------------|--------|----------------------|
| `hia`           | well absorbed              | poorly absorbed               | FP+GIN | HIA_Hou              |
| `bioavailability` | bioavailable (F≥20%)     | low oral bioavailability      | FP+GIN | Bioavailability_Ma   |
| `bbb`           | CNS penetrant              | non-CNS penetrant             | FP     | BBB_Martins          |
| `pgp`           | Pgp substrate (efflux)    | not a Pgp substrate           | FP+GIN | Pgp_Broccatelli      |
| `cyp1a2`        | CYP1A2 inhibitor           | not a CYP1A2 inhibitor        | FP+GIN | CYP1A2_Veith         |
| `cyp2c19`       | CYP2C19 inhibitor          | not a CYP2C19 inhibitor       | FP+GIN | CYP2C19_Veith        |
| `cyp2c9`        | CYP2C9 inhibitor           | not a CYP2C9 inhibitor        | FP+GIN | CYP2C9_Veith         |
| `cyp2d6`        | CYP2D6 inhibitor           | not a CYP2D6 inhibitor        | FP+GIN | CYP2D6_Veith         |
| `cyp3a4`        | CYP3A4 inhibitor           | not a CYP3A4 inhibitor        | FP+GIN | CYP3A4_Veith         |

All classification tasks use threshold **0.5** for the `class` label. The raw `probability` is always returned so callers can apply their own threshold.

**Model abbreviations**

| Code   | Full name                             | Features            |
|--------|---------------------------------------|---------------------|
| D-MPNN | Directed Message-Passing Neural Net   | molecular graph + RDKit 2D descriptors |
| FP+GIN | CatBoost + fingerprints + GIN embeddings | Morgan + Avalon + ErG + RDKit 2D + 300-dim GIN (2872-dim total) |
| FP     | CatBoost + fingerprints only          | Morgan + Avalon + ErG + RDKit 2D (2572-dim) |

---

## Error Responses

| HTTP status | When                                               | Body                                   |
|-------------|----------------------------------------------------|----------------------------------------|
| 400         | Missing `smiles` / `smiles_list`                   | `{"error": "'smiles' or 'smiles_list' required"}` |
| 400         | Batch > 1000 molecules                             | `{"error": "Batch limited to 1000 molecules per request"}` |
| 400         | All provided task keys are unknown                 | `{"error": "No valid tasks. Available: [...]"}` |
| 400         | GET `/predict` with no `?smiles=`                  | `{"error": "Provide ?smiles=<SMILES>"}` |
| 400         | Empty `smiles_list`                                | `{"error": "Empty input"}` |
| 500         | Internal prediction failure (bug / corrupted model) | `{"error": "Prediction error: <detail>"}` |
| 503         | RDKit or model files unavailable on startup        | `{"error": "Initialization error: <detail>"}` |

Note: an **invalid SMILES** that RDKit cannot parse is **not** a 400 error — it returns HTTP 200 with `"status": "invalid_smiles"` for that molecule inside `results`. This allows batches with some bad entries to still return results for the valid ones.

---

## Full Example

### Request

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'
```

### Response (aspirin)

```json
{
  "n_molecules": 1,
  "n_predicted": 1,
  "tasks": ["solubility","hia","caco2","bioavailability","bbb","pgp",
            "cyp1a2","cyp2c19","cyp2c9","cyp2d6","cyp3a4"],
  "elapsed_s": 3.821,
  "result": {
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "status": "ok",
    "predictions": {
      "solubility":      {"value": -1.7086, "unit": "log mol/L", "class": "freely soluble"},
      "hia":             {"probability": 0.9616, "class": "well absorbed"},
      "caco2":           {"value": -4.5802, "unit": "log cm/s",  "class": "high permeability (>7.1 nm/s)"},
      "bioavailability": {"probability": 0.7836, "class": "bioavailable (F≥20%)"},
      "bbb":             {"probability": 0.5975, "class": "CNS penetrant"},
      "pgp":             {"probability": 0.0235, "class": "not a Pgp substrate"},
      "cyp1a2":          {"probability": 0.4809, "class": "not a CYP1A2 inhibitor"},
      "cyp2c19":         {"probability": 0.2866, "class": "not a CYP2C19 inhibitor"},
      "cyp2c9":          {"probability": 0.0862, "class": "not a CYP2C9 inhibitor"},
      "cyp2d6":          {"probability": 0.0155, "class": "not a CYP2D6 inhibitor"},
      "cyp3a4":          {"probability": 0.0785, "class": "not a CYP3A4 inhibitor"}
    }
  },
  "results": [{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "...": "..."}]
}
```

### Python client example

```python
import requests

BASE = "http://localhost:5000"

# single molecule
r = requests.post(f"{BASE}/predict", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
data = r.json()
preds = data["result"]["predictions"]
print(f"logS = {preds['solubility']['value']:.2f}  ({preds['solubility']['class']})")
print(f"BBB  = {preds['bbb']['probability']:.2f}   ({preds['bbb']['class']})")

# batch — only absorption + CYP tasks
smiles_batch = [
    "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "c1ccc2ccccc2c1",           # naphthalene
]
r = requests.post(f"{BASE}/predict", json={
    "smiles_list": smiles_batch,
    "tasks": ["hia", "bioavailability", "cyp1a2", "cyp3a4"]
})
for mol in r.json()["results"]:
    if mol["status"] == "ok":
        p = mol["predictions"]
        print(f"{mol['canonical_smiles'][:30]:30s}  "
              f"HIA={p['hia']['probability']:.2f}  "
              f"CYP3A4={p['cyp3a4']['probability']:.2f}")
```

---

## Notes

- **Invalid SMILES in a batch** do not abort the request. Each molecule gets its own `status` field (`"ok"` or `"invalid_smiles"`). `n_predicted` reflects the count of molecules that succeeded.
- **`result` shorthand**: when the request contains exactly one molecule (either via `smiles` or a one-element `smiles_list`), the top-level response includes both `results[0]` and a `result` alias pointing to the same object.
- **Task filtering** via `"tasks"`: unknown task keys are silently dropped. If all keys are invalid a 400 is returned.
- **Thread safety**: the Flask development server is single-threaded by default. For concurrent use deploy behind gunicorn: `gunicorn -w 1 "api.app:app"` (keep 1 worker — models are held in process memory).
