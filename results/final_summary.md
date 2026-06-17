# ADME-T Final Results (best model per task)

| Task | Metric | Model | Ours (mean±std) | SOTA | Δ | Status |
|------|--------|-------|-----------------|------|---|--------|
| solubility | MAE↓ | dmpnn_rdkit | 0.7881±0.0044 | 0.761 | -0.0271 | ✗ gap |
| hia | AUROC↑ | FP+GIN | 0.9896±0.0007 | 0.989 | +0.0006 | ✅ at SOTA |
| caco2 | MAE↓ | FP+GIN | 0.2757±0.0065 | 0.256 | -0.0197 | ≈ near |
| bioavailability | AUROC↑ | FP+GIN | 0.7433±0.0069 | 0.938 | -0.1947 | ✗ gap |
| bbb | AUROC↑ | result | 0.9132±0.0041 | 0.916 | -0.0028 | ✅ at SOTA |
| pgp | AUROC↑ | FP+GIN | 0.9332±0.0026 | 0.938 | -0.0048 | ✅ at SOTA |
| cyp1a2 | AUROC↑ | FP+GIN | 0.9628±0.0007 | 0.930 | +0.0328 | ✅ at SOTA |
| cyp2c19 | AUROC↑ | FP+GIN | 0.9304±0.0004 | 0.900 | +0.0304 | ✅ at SOTA |
| cyp2c9 | AUPRC↑ | FP+GIN | 0.8582±0.0018 | 0.859 | -0.0008 | ✅ at SOTA |
| cyp2d6 | AUPRC↑ | FP+GIN | 0.7876±0.0020 | 0.790 | -0.0024 | ✅ at SOTA |
| cyp3a4 | AUPRC↑ | FP+GIN | 0.9156±0.0008 | 0.916 | -0.0004 | ✅ at SOTA |
