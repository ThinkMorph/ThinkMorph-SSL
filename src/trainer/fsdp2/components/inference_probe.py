from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceProbeSchedule:
    enabled: bool
    interval_steps: int
    start_step: int

    def should_run(self, step: int) -> bool:
        if not self.enabled:
            return False
        if step <= 0:
            return False
        if step < self.start_step:
            return False
        if self.interval_steps <= 0:
            return False
        return step % self.interval_steps == 0
