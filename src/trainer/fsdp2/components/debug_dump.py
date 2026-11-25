from __future__ import annotations


class DebugBatchDumper:
    def __init__(self, enabled: bool, interval_steps: int) -> None:
        self.enabled = enabled
        self.interval_steps = interval_steps

    def should_dump(self, global_step: int) -> bool:
        if not self.enabled:
            return False
        if self.interval_steps <= 0:
            return False
        return global_step % self.interval_steps == 0
