"""Dataset and processor registries.

Dataset and processor registries. Import from here instead.
"""

from loguru import logger


class DatasetRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type] = {}

    def register(self, name: str):
        def decorator(cls):
            if name in self._registry:
                logger.warning(
                    "{} already exists in dataset registry and will be overwritten.",
                    name,
                )
            self._registry[name] = cls
            return cls

        return decorator

    def __getitem__(self, key: str):
        if key not in self._registry:
            raise KeyError(
                f"Dataset '{key}' not registered. Available: {list(self._registry)}"
            )
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def get(self, key: str, default=None):
        return self._registry.get(key, default)

    def keys(self):
        return self._registry.keys()


class ProcessorRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type] = {}

    def register(self, name: str):
        def decorator(cls):
            if name in self._registry:
                logger.warning(
                    "{} already exists in processor registry and will be overwritten.",
                    name,
                )
            self._registry[name] = cls
            return cls

        return decorator

    def __getitem__(self, key: str):
        if key not in self._registry:
            raise KeyError(
                f"Processor '{key}' not registered. Available: {list(self._registry)}"
            )
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def get(self, key: str, default=None):
        return self._registry.get(key, default)


DATASET_REGISTRY = DatasetRegistry()
PROCESSOR_REGISTRY = ProcessorRegistry()

# Convenience aliases — same call signature as before: @register_dataset("name")
register_dataset = DATASET_REGISTRY.register
register_processor = PROCESSOR_REGISTRY.register

# Legacy dict-style access used by runner.py._build_dataset
DATASET_MAPPING = DATASET_REGISTRY._registry
DATAPROCESSOR_MAPPING = PROCESSOR_REGISTRY._registry
