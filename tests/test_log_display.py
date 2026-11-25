import importlib.util
from pathlib import Path

from loguru import logger


_LOG_DISPLAY_PATH = Path(__file__).resolve().parents[1] / "src" / "utils" / "log_display.py"
_SPEC = importlib.util.spec_from_file_location("log_display_under_test", _LOG_DISPLAY_PATH)
assert _SPEC is not None
log_display = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(log_display)
LogDisplay = log_display.LogDisplay


def test_update_preserves_zero_values_for_metric_fallbacks():
    messages = []
    sink_id = logger.add(lambda message: messages.append(message), format="{message}")
    try:
        display = LogDisplay(total_steps=1, enabled=False)
        display.update(
            1,
            {
                "train/loss": 0.0,
                "train/ce": 0.0,
                "train/mse": 0.0,
                "total_tokens": 0,
            },
        )
    finally:
        logger.remove(sink_id)

    step_messages = [str(message) for message in messages if "[step 1/1]" in str(message)]
    assert len(step_messages) == 1
    assert "loss=0.0000" in step_messages[0]
    assert "ce=0.0000" in step_messages[0]
    assert "mse=0.0000" in step_messages[0]
    assert "tokens=0" in step_messages[0]
