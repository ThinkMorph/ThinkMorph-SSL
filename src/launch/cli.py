import datetime
import os
from copy import deepcopy

import torch
import hydra
import torch.distributed as dist
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data import DatasetConfig
from eval import EvalConfig
from models import ModelConfig
from parallel.process_group_manager import setup_process_group_manager
from trainer import TrainerConfig, TrainingArguments, TrainRunner
from utils.distributed import (
    dist_barrier,
    install_mixed_pg_default_device_warning_filter,
)
from utils.logging_utils import setup_distributed_logging


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update nested dicts.

    This keeps unspecified defaults from `base` while allowing config_yaml to
    override only the keys it provides.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_dot_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply ``key=value`` overrides using dot-separated paths.

    Example overrides::

        ["trainer_args.learning_rate=2e-5", "trainer_args.run_name=exp1"]

    Values are parsed as YAML scalars so numbers, bools, and null are
    converted to the appropriate Python types automatically.
    """
    for item in overrides:
        if "=" not in item:
            logger.warning(f"Ignoring malformed override (no '='): {item}")
            continue
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
        logger.info(f"Override: {key} = {value!r}")
    return config


def _resolve_process_group_backend(use_cpu: bool) -> str:
    if use_cpu:
        return "gloo"
    # Enable both CPU and CUDA collectives so features like async checkpoint
    # save can use the CPU backend while training still uses NCCL on CUDA.
    return "cpu:gloo,cuda:nccl"


def create_train_task(config):
    logger.info("Parsing dataset/model/trainer configs...")
    dataset_config = config.pop("dataset_config")
    dataset_config = DatasetConfig(**dataset_config)

    model_config = config.pop("model_config")
    model_config = ModelConfig(**model_config)

    trainer_type = config.pop("trainer_type")
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    trainer_args = config.get("trainer_args")
    sp_degree = trainer_args.get("sp_ulysses_degree", 1)
    ep_degree = trainer_args.get("ep_degree", 1)
    # DP size actually will not be affected by ep_degree, but kept for initialization here
    dp_size = world_size // sp_degree

    # For now, we haven't implement the tp and pp
    use_cpu = trainer_args.get("use_cpu", False)
    backend = _resolve_process_group_backend(use_cpu=bool(use_cpu))
    # If the process group is already initialized, don't initialize it again
    ddp_timeout = trainer_args.get("ddp_timeout", 30 * 60)
    if not dist.is_initialized():
        logger.info(
            f"Initializing distributed process group: backend={backend}, rank={global_rank}, world_size={world_size}"
        )
        # For single GPU without distributed launcher, set required env vars
        if world_size == 1 and "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29601"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_RANK"] = "0"

        dist.init_process_group(
            rank=global_rank,
            world_size=world_size,
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(seconds=ddp_timeout),
        )
        logger.info("Distributed process group initialized.")
    install_mixed_pg_default_device_warning_filter()
    setup_process_group_manager(
        tp_size=1,
        cp_size=sp_degree,
        pp_size=1,
        dp_size=dp_size,
        ep_size=ep_degree,
    )
    logger.info("Process group manager initialized.")

    trainer_args = config.pop("trainer_args")

    eval_config_dict = trainer_args.pop("eval_config", None)
    if eval_config_dict is not None:
        eval_config = EvalConfig(**eval_config_dict)
        trainer_args["eval_config"] = eval_config.model_dump()

    trainer_args = TrainingArguments(**trainer_args)

    train_config = TrainerConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        trainer_type=trainer_type,
        trainer_args=trainer_args,
    )
    return TrainRunner(config=train_config)


def save_config(config):
    if dist.is_initialized():
        rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
    if rank == 0:
        trainer_args = config.get("trainer_args")
        output_dir = trainer_args.get("output_dir")
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    if dist.is_initialized():
        dist_barrier()


@hydra.main(version_base=None, config_path="config", config_name="default_config")
def main(config: DictConfig):
    torch.set_float32_matmul_precision("high")
    setup_distributed_logging()
    logger.info("Launcher started.")
    config = OmegaConf.to_yaml(config)
    config = yaml.safe_load(config)

    # If you have a predefined config yaml
    config_yaml = config.pop("config_yaml")
    if config_yaml:
        logger.info(
            f"Detected config yaml, merging with the default config. Will use the args in {config_yaml} to override current config."
        )
        with open(config_yaml, "r") as f:
            config_yaml = yaml.safe_load(f)
        _deep_update(config, config_yaml)

    # Apply dot-path overrides *after* config_yaml merge so they always win.
    # Pass as: bagel-train config_yaml=... +overrides='["k1=v1","k2=v2"]'
    # Or via env: BAGEL_OVERRIDES="k1=v1,k2=v2"
    cli_overrides = config.pop("overrides", None) or []
    env_overrides = os.environ.get("BAGEL_OVERRIDES", "")
    if env_overrides:
        cli_overrides.extend(env_overrides.split(","))
    if cli_overrides:
        _apply_dot_overrides(config, cli_overrides)

    original_config = deepcopy(config)
    logger.info("Creating train task...")
    task = create_train_task(config)
    logger.info("Saving merged config...")
    save_config(original_config)
    logger.info("Building training components...")
    task.build()
    logger.info("Starting training loop...")
    task.run()
    # Clean up trainer resources (DataLoader workers, compiled model caches,
    # async checkpoint threads) before destroying the process group.  Without
    # this the persistent DataLoader workers and/or Gloo threads from async
    # checkpoint saving can prevent dist.destroy_process_group() from completing.
    del task
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist_barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
