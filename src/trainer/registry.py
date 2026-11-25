from collections.abc import Callable
from typing import Any

from loguru import logger


class TrainerRegister:
    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., Any]] = {}

    def _add_item(self, key: str, value: Callable[..., Any]):
        if not callable(value):
            raise TypeError(f"Error: {value!r} must be callable.")
        if key in self._registry:
            logger.warning(
                "{} already exists in trainer registry and will be overwritten.",
                key,
            )
        self._registry[key] = value
        return value

    def register(self, target):
        if callable(target):
            return self._add_item(target.__name__, target)
        return lambda value: self._add_item(target, value)

    def __setitem__(self, key, value):
        self._registry[key] = value

    def __getitem__(self, key):
        return self._registry[key]

    def __contains__(self, key):
        return key in self._registry

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    def __str__(self):
        return str(self._registry)

    def get(self, key, default=None):
        return self._registry.get(key, default)

    def keys(self):
        return self._registry.keys()

    def values(self):
        return self._registry.values()

    def items(self):
        return self._registry.items()


TRAINER_REGISTER = TrainerRegister()
