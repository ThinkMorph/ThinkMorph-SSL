# ThinkMorph-SSL: Scaling Interleaved Multimodal Reasoning with Pretext Tasks

<p>
  <a href="https://arxiv.org/abs/2611.32183"><img src="https://img.shields.io/badge/arXiv-2611.32183-b31b1b?logo=arxiv&logoColor=white" alt="arXiv 2611.32183"/></a>
  <a href="https://thinkmorph.github.io/ssl/"><img src="https://img.shields.io/badge/Project-Page-13898a" alt="Project page"/></a>
  <a href="https://huggingface.co/ThinkMorph/ThinkMorph-SSL-Instruct"><img src="https://img.shields.io/badge/ThinkMorph--SSL--Instruct-Model-orange?logo=huggingface&logoColor=orange" alt="ThinkMorph-SSL-Instruct Model"/></a>
  <a href="https://huggingface.co/ThinkMorph/ThinkMorph-SSL-Base"><img src="https://img.shields.io/badge/ThinkMorph--SSL--Base-Model-orange?logo=huggingface&logoColor=orange" alt="ThinkMorph-SSL-Base Model"/></a>
  <a href="https://huggingface.co/datasets/ThinkMorph/Pretext-270K"><img src="https://img.shields.io/badge/Pretext--270K-Dataset-yellow?logo=huggingface&logoColor=yellow" alt="Pretext-270K Dataset"/></a>
  <a href="https://huggingface.co/datasets/ThinkMorph/Instruct-180K"><img src="https://img.shields.io/badge/Instruct--180K-Dataset-yellow?logo=huggingface&logoColor=yellow" alt="Instruct-180K Dataset"/></a>
</p>

Training and evaluation code for ThinkMorph-SSL.

## News

- **[2026.11.24]** The paper is on [arXiv](https://arxiv.org/abs/2611.32183), and the project page is live at [thinkmorph.github.io/ssl](https://thinkmorph.github.io/ssl/).

## About

ThinkMorph-SSL reasons with interleaved text and images. On a visual question it thinks in text,
generates intermediate images when they help (a zoomed-in crop, a depth map, a bird's-eye view,
the view from another position), and then answers.

It is trained from BAGEL-7B-MoT in two stages:

| Stage | Data | Config | Model |
| --- | --- | --- | --- |
| 1. Self-supervised pretext reasoning | [Pretext-270K](https://huggingface.co/datasets/ThinkMorph/Pretext-270K) | `configs/train/stage1_pretext.yaml` | [ThinkMorph-SSL-Base](https://huggingface.co/ThinkMorph/ThinkMorph-SSL-Base) |
| 2. Instruction tuning | [Instruct-180K](https://huggingface.co/datasets/ThinkMorph/Instruct-180K) | `configs/train/stage2_instruct.yaml` | [ThinkMorph-SSL-Instruct](https://huggingface.co/ThinkMorph/ThinkMorph-SSL-Instruct) |

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/). `flash-attn` is built from source, so `uv sync`
needs a CUDA toolkit: `nvcc` on `PATH` and `CUDA_HOME` set.

```bash
git clone https://github.com/ThinkMorph/ThinkMorph-SSL.git
cd ThinkMorph-SSL
uv sync --frozen
source .venv/bin/activate
uv pip install -e eval/VLMEvalKit
```

`secrets.env.example` lists the paths and keys the scripts read. Copy it to `secrets.env`, fill it
in, and `source` it.

## Data and models

The [data guide](data/README.md) covers download and layout. Download the base model and, if you
only want to evaluate, the released model:

```bash
hf download lmms-lab/BAGEL-7B-MoT-ver.LE --local-dir models/BAGEL-7B-MoT-ver.LE
hf download ThinkMorph/ThinkMorph-SSL-Instruct --local-dir models/ThinkMorph-SSL-Instruct
```

## Training

```bash
export BAGEL_MODEL_DIR=$PWD/models/BAGEL-7B-MoT-ver.LE
export DATA_ROOT=/path/to/data

# Stage 1. Run on every node, each with its own NODE_RANK.
NGPUS=8 NNODES=4 NODE_RANK=0 MASTER_ADDR=<rank-0-address> \
  bash scripts/train.sh configs/train/stage1_pretext.yaml

# Merge the EMA weights of Stage 1.
python scripts/merge_checkpoint.py \
  outputs/stage1_pretext/checkpoints/checkpoint-5500 \
  outputs/stage1_pretext/merged \
  "$BAGEL_MODEL_DIR" --checkpoint-type ema

# Stage 2. To skip Stage 1, point STAGE1_CKPT at a copy of ThinkMorph-SSL-Base.
export STAGE1_CKPT=$PWD/outputs/stage1_pretext/merged
NGPUS=8 NNODES=6 NODE_RANK=0 MASTER_ADDR=<rank-0-address> \
  bash scripts/train.sh configs/train/stage2_instruct.yaml
```

The commands above use the original setup: 32 GPUs for Stage 1 and 48 for Stage 2. Batches are
packed dynamically, so a different number of GPUs changes how much data each step covers. Extra
`key=value` arguments to `scripts/train.sh` override config values.

Both stages use a cosine learning-rate schedule with 1000 warmup steps, defined over `max_steps` (10000 for Stage 1, 12000 for Stage 2).
The released checkpoints are the EMA weights at step 5500 of each run, so you can stop a run once `checkpoint-5500` is saved.
Keep `max_steps` as it is: setting it to 5500 makes the learning rate decay to zero by step 5500, which trains a different model.
The configs keep enough checkpoints that step 5500 is not deleted if a run goes to the end.

The paper runs also trained on the rows whose images come from Matterport3D and Ego-Exo4D.
Those rows ship without images (see [Rows without images](data/README.md#rows-without-images)), and their folders are commented out in the configs.
Without them you train on a subset of the paper's data.

`configs/train/ablation_mindcube/` holds the MindCube staging experiment. Its Stage 1 data is not
released.

## Evaluation

```bash
export SSL_MODEL_DIR=$PWD/models/ThinkMorph-SSL-Instruct
export LMUData=/path/to/benchmark/data
export OPENAI_API_KEY=<key>        # judge; set OPENAI_API_BASE for a compatible endpoint
EVAL_NGPUS=8 bash eval/run_eval.sh configs/eval/ssl.yaml
```

`configs/eval/ssl.yaml` lists the benchmarks and the inference settings. Each benchmark is one
job; a rerun skips finished ones. Results and the reasoning images go to
`outputs/eval/thinkmorph_ssl_instruct/`. The evaluator is a fork of
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit), in `eval/VLMEvalKit/`.

VLMEvalKit downloads each benchmark's TSV into `$LMUData` the first time it runs, so the directory can start empty.

The paper's numbers use these inference settings:

- **Text decoding:** greedy, with up to 16384 thinking tokens in total and up to 1024 tokens in each text segment between generated images. `text_temperature` only takes effect when `do_sample` is true.
- **Image inputs:** the ViT sees images through `ImageTransform(980, 224, 14)`, which is BAGEL's original setting, and the VAE through `ImageTransform(1024, 512, 16)`.
- **Image generation:** 50 denoising steps, `cfg_text_scale` 3.0, `cfg_img_scale` 1.5, `cfg_interval` [0.4, 1.0], global CFG renormalization with minimum 1.0, and `timestep_shift` 3.0. The model declares each image's size; 512 x 512 is the fallback.

Only the decoding settings are exposed in the config.
The image settings are fixed in `eval/VLMEvalKit/vlmeval/vlm/ThinkMorphInterleaved.py`.

The judge is GPT-5 (`gpt-5`).
MMSI, MindCube, OmniSpatial and SpatialViz grade every response with it, and the other benchmarks call it only when rule-based option matching fails.
If the judge is unreachable, the spatial evaluator falls back to rule-based scoring with a warning, and the scores change.
Check the log for `fallback to rule-based` before reporting numbers.

## License

The code is under [Apache-2.0](LICENSE). Some vendored files keep their own terms, including
DiT-derived utilities under CC BY-NC 4.0; see [third-party notices](THIRD_PARTY_NOTICES.md). Models
and datasets carry their own terms, stated on their cards.

## Citation

```bibtex
@article{gu2026thinkmorphssl,
  title   = {ThinkMorph-SSL: Scaling Interleaved Multimodal Reasoning with Pretext Tasks},
  author  = {Gu, Jiawei and Li, Linjie and Liu, Yiming and Hao, Yunzhuo and Wang, Huichen and
             Chen, Guanzheng and Peng, Zhichao and Xu, Luxin and Zhang, Xinyu and Luo, Li and
             Lan, Disen and Hu, Zican and Song, Mingyang and Valente, David and Wang, Alex Jinpeng and
             Li, Yafu and Cui, Ganqu and Yang, Zhengyuan and Xie, Qizhe and Choi, Yejin and
             Krishna, Ranjay and Cheng, Yu},
  journal = {arXiv preprint arXiv:2611.32183},
  year    = {2026}
}
```
