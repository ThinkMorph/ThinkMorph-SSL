# VLMEvalKit fork for ThinkMorph-SSL

This is a fork of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) used to evaluate ThinkMorph-SSL and the BAGEL baselines in the paper. It adds:

1. `vlmeval/vlm/ThinkMorphInterleaved.py`: interleaved text and image decoding for ThinkMorph-SSL checkpoints. It imports the model and inferencer from the repository's `src/`, so evaluation runs the same model code as training.
2. `vlmeval/vlm/Bagel.py`: BAGEL-7B-MoT text-only baselines (no-think, think-4k, think-16k).
3. Dataset modules for MMSI, MindCube, OmniSpatial, and SpatialViz-Bench.

Use `eval/run_eval.sh` at the repository root to evaluate a ThinkMorph-SSL checkpoint. It registers the checkpoint at runtime through `SSL_*` environment variables (see the end of the BAGEL section in `vlmeval/config.py`) and calls `run.py` once per benchmark.

The upstream documentation is in `docs/`. VLMEvalKit is released under the Apache 2.0 license (see `LICENSE`).
