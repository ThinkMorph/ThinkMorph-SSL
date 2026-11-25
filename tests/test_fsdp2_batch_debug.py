from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainer.fsdp2.fsdp2_trainer import FSDP2SFTTrainer


class _FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.mapping[token_id] for token_id in token_ids)


def test_format_token_context_windows_highlights_action_token():
    trainer = FSDP2SFTTrainer.__new__(FSDP2SFTTrainer)
    tokenizer = _FakeTokenizer(
        {
            10: "<|im_start|>",
            11: "<think>",
            12: "plan",
            13: "</think>",
            99: "<gen_image>",
            14: "(512,512)",
            15: "<|im_end|>",
        }
    )
    token_ids = torch.tensor([10, 11, 12, 13, 99, 14, 15], dtype=torch.long)

    lines = trainer._format_token_context_windows(
        token_ids=token_ids,
        tokenizer=tokenizer,
        target_token_id=99,
        context_tokens=2,
        label="action token",
    )

    assert lines[0] == "--- action token contexts (+/-2 tokens) ---"
    assert "[hit 0 @ 4]" in lines[1]
    assert "<<<" in lines[1]
    assert "<gen_image>" in lines[1]
    assert "(512,512)" in lines[1]
