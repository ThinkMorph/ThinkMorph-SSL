# Data

Training uses two public datasets. Each is organized by task, where a task is a visual operation:
what the intermediate images do. The dataset cards list the tasks and each source's license.

| Dataset | Rows | Tasks |
| --- | ---: | ---: |
| [Pretext-270K](https://huggingface.co/datasets/ThinkMorph/Pretext-270K) | 270,896 train, 1,600 val | 7 |
| [Instruct-180K](https://huggingface.co/datasets/ThinkMorph/Instruct-180K) | 180,490 | 17 |

## Download

```bash
export DATA_ROOT=/path/to/data

hf download ThinkMorph/Pretext-270K --repo-type dataset --include "shards/*" --local-dir Pretext-270K
mkdir -p "$DATA_ROOT/ssl-pretext-270k"
for f in Pretext-270K/shards/*.tar; do tar -xf "$f" -C "$DATA_ROOT/ssl-pretext-270k"; done

hf download ThinkMorph/Instruct-180K --repo-type dataset --include "shards/*" --local-dir Instruct-180K
mkdir -p "$DATA_ROOT/ssl-instruct-180k"
for f in Instruct-180K/shards/*.tar; do tar -xf "$f" -C "$DATA_ROOT/ssl-instruct-180k"; done
```

To check every archive and extracted file against its recorded SHA256, restore with the script
instead of `tar`:

```bash
python scripts/shard_data_release.py restore \
  Instruct-180K/shards/SHARDS.json Instruct-180K/shards "$DATA_ROOT/ssl-instruct-180k"
```

## Layout

```text
DATA_ROOT/
  ssl-pretext-270k/
    MANIFEST.json
    images/<sha256>.<ext>        shared by all tasks
    colorization/train.jsonl     image paths in rows: ../images/<sha256>.<ext>
    colorization/val.jsonl
    ...                          one folder per task
  ssl-instruct-180k/
    MANIFEST.json
    images/<sha256>.<ext>
    bbox/train.jsonl
    ...
```

The training configs list one folder per task. Stage 1 reads `train.jsonl`.

## Row format

```json
{"id": "<sha256>", "task": "bbox", "question": "...", "images": ["../images/<sha256>.jpg"],
 "reasoning_thought_0": "...", "reasoning_image_0": "../images/<sha256>.jpg",
 "reasoning_thought_1": "...", "answer": "B"}
```

A trace reads in this order: the input images and the question, then `reasoning_thought_0`,
`reasoning_image_0`, `reasoning_thought_1`, and so on, then the answer. Image paths are relative
to the row's file. Pretext rows have no question or answer. Stage 2 needs a first thought, a first
reasoning image and an answer on every row. To train on your own data, write rows in this format.

## Rows without images

Rows whose images come from Matterport3D or Ego-Exo4D are in `text_only/` of each dataset, with
the same task folders. The dataset cards explain how to obtain the images. To train on them, put
`text_only/` at `$DATA_ROOT/ssl-instruct-180k-text-only` (or `ssl-pretext-270k-text-only`),
add the images under its `images/` folder, and uncomment those folders in the training config.

[`builders/egoexo/`](builders/egoexo/README.md) regenerates the Ego-Exo4D frames from your own
copy of that dataset.
