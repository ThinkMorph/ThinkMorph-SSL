from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.bagel.bagel import _convert_bool_attention_masks_to_float


def test_convert_bool_attention_masks_to_float_reuses_list_and_replaces_entries():
    masks = [
        torch.tensor([[True, False], [True, True]]),
        torch.tensor([[False, True], [True, False]]),
    ]

    converted = _convert_bool_attention_masks_to_float(masks, torch.device("cpu"))

    assert converted is masks
    assert converted[0].dtype == torch.float
    assert converted[1].dtype == torch.float
    assert torch.isneginf(converted[0][0, 1])
    assert converted[0][0, 0].item() == 0.0
