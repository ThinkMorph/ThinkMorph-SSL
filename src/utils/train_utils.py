import math
from typing import Union

import torch
from loguru import logger


class TrainUtilities:
    @staticmethod
    def format_tokens(tokens: Union[int, float]) -> str:
        """Format a token count into a compact string with unit K/M/B/T.

        - Uses a 1,000-based scale (K=1e3, M=1e6, B=1e9, T=1e12)
        - Produces a short, readable number (up to ~3 significant digits)
        - Trims trailing zeros and the decimal point when unnecessary

        Examples:
            532 -> "532"
            12_345 -> "12.3K"
            1_234_567 -> "1.23M"
            98_700_000 -> "98.7M"
            12_300_000_000 -> "12.3B"
            1_234_000_000_000 -> "1.23T"

        Args:
            tokens: Total number of tokens (can be int or float)

        Returns:
            A compact string representation with an appropriate unit suffix.
        """
        # Handle non-finite and None-like inputs gracefully
        if tokens is None:
            return "0"
        try:
            value = float(tokens)
        except (TypeError, ValueError):
            return str(tokens)

        if not math.isfinite(value):
            return str(tokens)

        sign = "-" if value < 0 else ""
        value = abs(value)

        units = ["", "K", "M", "B", "T"]
        idx = 0
        while value >= 1000.0 and idx < len(units) - 1:
            value /= 1000.0
            idx += 1

        # Choose decimals to keep around 3 significant digits
        if value >= 100:
            decimals = 0
        elif value >= 10:
            decimals = 1
        else:
            decimals = 2

        formatted = f"{value:.{decimals}f}"
        # Strip trailing zeros and unnecessary decimal point
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")

        return f"{sign}{formatted}{units[idx]}"

    @staticmethod
    def convert_open_to_hf(messages):
        hf_messages = []
        for message in messages:
            new_message = {"role": message["role"], "content": []}
            for content in message["content"]:
                if content["type"] == "image_url":
                    new_message["content"].append({"type": "image"})
                elif content["type"] == "audio_url":
                    new_message["content"].append(
                        {"type": "audio", "audio_url": content["audio_url"]["url"]}
                    )
                elif content["type"] == "video_url":
                    new_message["content"].append(
                        {"type": "video", "video_url": content["video_url"]["url"]}
                    )
                else:
                    new_content = {"type": "text", "text": content["text"]}
                    if "audio_text" in content:
                        new_content["audio_text"] = content["audio_text"]
                    new_message["content"].append(new_content)
            hf_messages.append(new_message)

        return hf_messages

    @staticmethod
    def get_device_flops(unit="T"):
        def unit_convert(number, level):
            units = ["B", "K", "M", "G", "T", "P"]
            if number <= 0:
                return number
            ptr = 0
            while ptr < len(units) and units[ptr] != level:
                number /= 1000
                ptr += 1
            return number

        device_name = torch.cuda.get_device_name()
        flops = float("inf")  # INF flops for unknown gpu type

        if "MI300X" in device_name:
            flops = 1336e12
        elif "H100" in device_name or "H800" in device_name or "H200" in device_name:
            flops = 989e12
        elif "A100" in device_name or "A800" in device_name:
            flops = 312e12
        elif "L40" in device_name:
            flops = 181.05e12
        elif "L20" in device_name:
            flops = 119.5e12
        elif "H20" in device_name:
            flops = 148e12
        elif "910B" in device_name:
            flops = 354e12
        elif "RTX 3070 Ti" in device_name:
            flops = 21.75e12
        flops_unit = unit_convert(flops, unit)
        return flops_unit

    @staticmethod
    def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
        to_return = {
            k: t
            for k, t in named_params
            if any(key_match in k for key_match in keys_to_match)
        }
        to_return = {
            k: TrainUtilities.maybe_zero_3(v, ignore_status=True).cpu()
            for k, v in to_return.items()
        }
        return to_return

    @staticmethod
    def maybe_zero_3(param, ignore_status=False, name=None):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        if hasattr(param, "ds_id"):
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                if not ignore_status:
                    logger.warning(
                        f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                    )
            with zero.GatheredParameters([param]):
                param = param.data.detach().cpu().clone()
        else:
            param = param.detach().cpu().clone()
        return param
