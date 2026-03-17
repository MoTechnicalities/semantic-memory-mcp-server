# RWIF vs Dense Cosine on a Larger Shipped Text Corpus

## Summary

- Dataset kind: `real-text-hashed-corpus`
- Records: `24`
- Queries: `12`
- Calibration texts: `8`
- Vector length: `384`
- Top-k wave units per record: `96`
- RWIF wave top-1 accuracy: `100.0%`
- Dense cosine top-1 accuracy: `100.0%`
- Top-1 agreement rate: `100.0%`
- Full-ranking agreement rate: `0.0%`
- Mean Spearman rank correlation: `0.813`
- Mean absolute rank shift: `2.972`
- Top-1 overlap rate: `100.0%`
- Top-3 overlap rate: `75.0%`
- Top-5 overlap rate: `73.3%`
- RWIF store artifact size: `44,260` bytes
- Dense float32 matrix estimate: `36,864` bytes
- Dense float64 matrix estimate: `73,728` bytes
- RWIF vs dense float32 size ratio: `1.201`
- RWIF vs dense float64 size ratio: `0.600`

## Per-Query Snapshot

| Query | Expected | RWIF top | Cosine top |
|---|---|---|---|
| Which capital city in France is crossed by the Seine? | geo-paris | geo-paris | geo-paris |
| What city is the capital of Egypt and sits on the Nile River? | geo-cairo | geo-cairo | geo-cairo |
| Which router feature keeps a printer on the same local IP address? | net-dhcp | net-dhcp | net-dhcp |
| What wireless setting lets visitors use the internet without reaching private home devices? | net-guest | net-guest | net-guest |
| What kind of semantic store may answer queries but must reject writes? | rwif-readonly | rwif-readonly | rwif-readonly |
| What config combines several semantic stores behind one MCP surface? | rwif-federated | rwif-federated | rwif-federated |
| How do you run Python's built-in test discovery from the command line? | py-unittest | py-unittest | py-unittest |
| Which file format keeps one JSON object per line for streaming records? | py-jsonl | py-jsonl | py-jsonl |
| What eclipse occurs when Earth's shadow falls across the Moon? | astro-eclipse | astro-eclipse | astro-eclipse |
| Why do comet tails point away from the Sun? | astro-comet | astro-comet | astro-comet |
| Which herb prefers well-drained soil and moderate watering? | garden-rosemary | garden-rosemary | garden-rosemary |
| What garden covering helps the soil hold moisture and block weeds? | garden-mulch | garden-mulch | garden-mulch |

## Interpretation

This benchmark scales the shipped workload comparison up to a larger text corpus and uses deterministic hashed-text activations so the full benchmark remains reproducible with the base runtime dependencies only.

## Caveat

The activations here are deterministic text hashes, not a learned embedding model. This benchmark is intended as a larger reproducible release check for RWIF-versus-dense behavior, not as a substitute for transformer-backed evaluation.
