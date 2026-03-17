# RWIF vs Dense Cosine Baseline

## Summary

- Dataset kind: `synthetic-sparse-spectrum`
- Records: `64`
- Queries: `16`
- Vector length: `1024`
- Top-k wave units per record: `128`
- RWIF compressed top-1 accuracy: `100.0%`
- Dense cosine top-1 accuracy: `100.0%`
- Agreement rate: `100.0%`
- RWIF file size: `143,320` bytes
- Dense float32 matrix estimate: `262,144` bytes
- Dense float64 matrix estimate: `524,288` bytes
- RWIF vs dense float32 size ratio: `0.547`
- RWIF vs dense float64 size ratio: `0.273`

## Interpretation

This benchmark compares one concrete vector baseline against the RWIF storage method: brute-force dense cosine retrieval on original activations versus cosine retrieval on RWIF-compressed reconstructions of those same activations.
The storage comparison only models the embedding matrix itself. It does not include external database services, index structures, metadata services, or orchestration overhead.

## Caveat

This is a synthetic sparse-spectrum benchmark designed to isolate the file-storage implications of the RWIF encoding. It is not a claim about all embedding distributions or all vector database systems.
