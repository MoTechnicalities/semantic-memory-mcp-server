# RWIF vs Dense Cosine on a Real Text Array Dataset

## Summary

- Dataset kind: `real-text-array`
- Records: `6`
- Queries: `3`
- Calibration texts: `4`
- Vector length: `4`
- Top-k wave units per record: `4`
- RWIF wave top-1 accuracy: `100.0%`
- Dense cosine top-1 accuracy: `100.0%`
- Top-1 agreement rate: `100.0%`
- Full-ranking agreement rate: `100.0%`
- Mean Spearman rank correlation: `1.000`
- Mean absolute rank shift: `0.000`
- Top-1 overlap rate: `100.0%`
- Top-3 overlap rate: `100.0%`
- RWIF store artifact size: `2,433` bytes

## Per-Query Snapshot

| Query | Expected | RWIF top | Cosine top |
|---|---|---|---|
| Rome is the capital of Italy. | geo-paris | geo-paris | geo-paris |
| One plus two makes three. | math-addition | math-addition | math-addition |
| The river flows between the hills. | nature-river | nature-river | nature-river |

## Interpretation

This benchmark is workload-shaped rather than storage-shaped. It uses real text records, real text queries, a shipped calibration set, the RWIF wave retrieval path, and the dense cosine baseline on the same text-derived activations.

## Caveat

The dataset is intentionally tiny and array-backed so it can ship in the public repository as a deterministic release check. It should be read as a reproducible parity benchmark, not as a large-corpus performance claim.
