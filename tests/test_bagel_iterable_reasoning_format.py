from pathlib import Path
from types import SimpleNamespace
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.iterable.bagel_iterable_dataset import BagelIterableDataset
from data.processor.config import ProcessorConfig


def _make_dataset(*, supervise_image_action_token: bool):
    dataset = BagelIterableDataset.__new__(BagelIterableDataset)
    dataset.config = SimpleNamespace(
        jsonl_images_root="data/jigsaw_assembly",
        jsonl_reasoning_images_root="data/jigsaw_assembly",
        use_action_control_system_prompt=False,
        action_control_system_prompt=None,
    )
    dataset.processor_config = ProcessorConfig(
        processor_name="dummy",
        processor_type="bagel",
        extra_kwargs={
            "supervise_image_action_token": supervise_image_action_token,
        },
    )
    dataset.wrap_assistant_reasoning_with_think = True
    dataset.use_action_control_system_prompt = False
    dataset.action_control_system_prompt = None
    return dataset


def _sample_row():
    return {
        "question": "Assemble the puzzle.",
        "images": ["images/input.png"],
        "reasoning_thought_0": "inspect seam",
        "reasoning_image_0": "reasoning_images/step0.png",
        "reasoning_thought_1": "verify scene",
        "gt_answer": "(B) Part 1 should be to the left of Part 2",
    }


def test_reasoning_jsonl_messages_use_action_token_friendly_format_when_supervised():
    dataset = _make_dataset(supervise_image_action_token=True)

    result = dataset._build_reasoning_jsonl_messages(_sample_row())
    assistant_content = result["messages"][1]["content"]

    assert [item["type"] for item in assistant_content] == ["text", "image_url", "text"]
    assert assistant_content[0]["text"] == "<think>inspect seam</think>"
    assert assistant_content[2]["text"] == (
        "<think>verify scene</think>"
        "<answer>(B) Part 1 should be to the left of Part 2</answer>"
    )
    assert "<image_start>" not in assistant_content[0]["text"]
    assert "<image_end>" not in assistant_content[2]["text"]


def test_reasoning_jsonl_messages_without_action_token_uses_same_format():
    """After removing the legacy <image_start>/<image_end> path, both
    action-token and non-action-token configs produce the same output."""
    dataset = _make_dataset(supervise_image_action_token=False)

    result = dataset._build_reasoning_jsonl_messages(_sample_row())
    assistant_content = result["messages"][1]["content"]

    assert [item["type"] for item in assistant_content] == ["text", "image_url", "text"]
    assert assistant_content[0]["text"] == "<think>inspect seam</think>"
    assert assistant_content[2]["text"] == (
        "<think>verify scene</think>"
        "<answer>(B) Part 1 should be to the left of Part 2</answer>"
    )
    assert "<image_start>" not in assistant_content[0]["text"]
    assert "<image_end>" not in assistant_content[2]["text"]


def test_reasoning_jsonl_messages_appends_options_to_question():
    dataset = _make_dataset(supervise_image_action_token=True)
    row = _sample_row()
    row["question"] = (
        "Treat the projected screen as being west of the red wall lights. "
        "In which direction are the floor candles relative to the red wall lights?"
    )
    row["options"] = "A: West, B: South, C: East, D: North"

    result = dataset._build_reasoning_jsonl_messages(row)

    assert result["messages"][0]["content"][1]["text"] == (
        "Treat the projected screen as being west of the red wall lights. "
        "In which direction are the floor candles relative to the red wall lights?\n"
        "Options: A: West, B: South, C: East, D: North"
    )


def test_reasoning_jsonl_messages_keeps_question_when_options_are_empty():
    dataset = _make_dataset(supervise_image_action_token=True)
    row = _sample_row()
    row["options"] = "  "

    result = dataset._build_reasoning_jsonl_messages(row)

    assert result["messages"][0]["content"][1]["text"] == "Assemble the puzzle."


def test_reasoning_jsonl_messages_does_not_duplicate_existing_options_text():
    dataset = _make_dataset(supervise_image_action_token=True)
    row = _sample_row()
    row["question"] = "Choose the direction.\nOptions: A: West, B: East"
    row["options"] = "A: West, B: East"

    result = dataset._build_reasoning_jsonl_messages(row)

    assert result["messages"][0]["content"][1]["text"] == (
        "Choose the direction.\nOptions: A: West, B: East"
    )
