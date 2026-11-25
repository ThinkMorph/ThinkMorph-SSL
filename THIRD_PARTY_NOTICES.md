# Third-party components

The project-level Apache-2.0 license does not replace the licenses attached to individual vendored files, model weights, or source datasets.

| Component | Included location | Upstream and retained terms |
| --- | --- | --- |
| BAGEL model implementation | `src/models/bagel/`; evaluation wrappers' `modeling/` trees | [BAGEL](https://github.com/ByteDance-Seed/Bagel), with original ByteDance notices; most files identify Apache-2.0 |
| DiT-derived position/timestep utilities | `src/models/bagel/modeling_utils.py`; `eval/VLMEvalKit/vlmeval/vlm/{bagel,thinkmorph}/modeling/bagel/modeling_utils.py` | File headers identify CC BY-NC 4.0. The [DiT license](https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt) is included as `licenses/DiT-CC-BY-NC-4.0.txt`; the noncommercial condition is retained |
| Flux-derived autoencoder | BAGEL `autoencoder.py` copies | Retain Black Forest Labs and ByteDance Apache-2.0 notices |
| Qwen2 and SigLIP implementations | BAGEL `qwen2`, `siglip`, and corresponding evaluation copies | Retain Qwen/Hugging Face/ByteDance source notices and Apache-2.0 terms |
| VLMEvalKit | `eval/VLMEvalKit/` | [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), Apache-2.0; its own `LICENSE` and source notices remain included |

The bundled DiT-derived files were adapted upstream and may contain subsequent project changes. Retain their original modification notices. No relicensing of those files is asserted by the top-level license.

Model and dataset cards state their own terms. Images in the datasets keep the terms of the sources they come from; this repository does not assign a license to them.
