# Ego-Exo4D frame export (Instruct-180K `exo_to_ego`)

The `exo_to_ego` task is released as text only: the Ego-Exo4D license does not allow its frames
in a dataset. This script is the frame selection used to build that task. With your own
copy of Ego-Exo4D it regenerates the frames the rows refer to.

## What it selects

For each take:

- **Ego frames.** It samples the ego (Aria) video every `--sample-step` frames inside the
  take's time-synced range. It scores each sample by the Laplacian variance of the rectified
  ego center (`--score-size` px). It keeps the `--top-k` best samples that are at least
  `--min-gap` frames apart.
- **Exo frames.** The exo frame is the frame-aligned frame of the take's `best_exo` camera,
  read from Ego-Exo4D `takes.json`. When that field or its video is missing, it uses `cam01`.
- **Output.** It writes `ego_<rank>.jpg`, `exo_<rank>.jpg`, `exo_marked_<rank>.jpg` and
  `metadata.json`, which records `frame_idx`, `time_sec_take_video`, `global_time` and
  `best_exo`.

A row image named `egoexo_take_ninepack_all__<take>__ego_<rank>__<tag>.jpg` is
`<take>/ego_<rank>.jpg` (the same holds for `exo`). The trailing tag is an
identifier, not a content hash. Check a regenerated frame against the size and MD5 recorded
in `text_only/images.jsonl`.

## Parameters used for the release

```bash
python export_take_ninepack.py --dataset-root /path/to/egoexo --out-dir egoexo_take_ninepack_all \
  --all-takes --top-k 3 --min-gap 120 --sample-step 128 --crop-ratio 0 \
  --marker-radius 64 --marker-thickness 8 --marker-cross-size 96 --score-size 320
```

Options not listed kept the script defaults: `--ego-rectify-mode approximate` and
`--ego-size 1024`. Byte-identical output also depends on the video decoder (OpenCV) build.
