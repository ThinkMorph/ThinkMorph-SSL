"""In-memory adapter that wraps a training model as a VLMEvalKit-compatible model.

Uses the training codebase's InterleaveInferencer (not VLMEvalKit's) to ensure
inference consistency with training probes.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

# VLMEvalKit base class


class ThinkMorphInMemory:
    """VLMEvalKit-compatible model wrapper for in-memory training models.

    Implements the subset of VLMEvalKit's BaseModel interface needed by the
    evaluation pipeline (generate, generate_inner) without inheriting from
    BaseModel to avoid import-time side effects.
    """

    INTERLEAVE = True
    INSTALL_REQ = False
    allowed_types = ["text", "image"]

    def __init__(
        self,
        model: torch.nn.Module,
        vae_model: torch.nn.Module,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids: dict,
        infer_kwargs: Dict[str, Any],
        understanding_output: bool = True,
        save_dir: Optional[str] = None,
    ):
        from models.bagel.inferencer import InterleaveInferencer

        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        self.infer_kwargs = dict(infer_kwargs)
        self.understanding_output = understanding_output
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def _build_input_list(self, message: List[Dict[str, Any]]) -> list:
        """Convert VLMEvalKit message format to InterleaveInferencer input list."""
        input_list = []
        for m in message:
            if m["type"] == "text":
                input_list.append(m["value"])
            elif m["type"] == "image":
                val = m["value"]
                if isinstance(val, str):
                    img = Image.open(val).convert("RGB")
                elif isinstance(val, Image.Image):
                    img = val
                else:
                    raise TypeError(f"Unsupported image input type {type(val)}")
                input_list.append(img)
            else:
                raise ValueError(f"Unsupported message type {m['type']}")
        return input_list

    def _extract_output(self, output_list: list) -> str:
        """Convert InterleaveInferencer output to a string for VLMEvalKit."""
        if self.understanding_output:
            return output_list[0] if output_list else ""

        results = []
        text_round = 0
        for idx, out_item in enumerate(output_list):
            if isinstance(out_item, str):
                out_item = f"Round_{text_round}:\n" + out_item
                results.append(out_item)
                text_round += 1
            elif isinstance(out_item, Image.Image):
                if self.save_dir is not None:
                    out_img_path = os.path.join(
                        self.save_dir,
                        f"bench_eval_{uuid.uuid4().hex[:8]}_{idx}.jpg",
                    )
                    out_item.save(out_img_path)
                    results.append(f"[Image: {out_img_path}]")
                else:
                    results.append("[Image: <not saved>]")
        return "\n".join(results)

    def generate_inner(self, message, dataset=None):
        input_list = self._build_input_list(message)
        with torch.no_grad():
            output_list = self.inferencer.interleave_inference(
                input_list,
                understanding_output=self.understanding_output,
                **self.infer_kwargs,
            )
        return self._extract_output(output_list)

    # --- Minimal BaseModel interface for VLMEvalKit compatibility ---

    def check_content(self, msgs):
        if isinstance(msgs, str):
            return "str"
        if isinstance(msgs, dict):
            return "dict"
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == "str" for t in types):
                return "liststr"
            if all(t == "dict" for t in types):
                return "listdict"
        return "unknown"

    def preproc_content(self, inputs):
        if self.check_content(inputs) == "str":
            return [dict(type="text", value=inputs)]
        if self.check_content(inputs) == "dict":
            return [inputs]
        if self.check_content(inputs) == "listdict":
            return inputs
        if self.check_content(inputs) == "liststr":
            res = []
            for s in inputs:
                res.append(dict(type="text", value=s))
            return res
        return None

    def generate(self, message, dataset=None):
        assert self.check_content(message) in [
            "str",
            "dict",
            "liststr",
            "listdict",
        ], f"Invalid input type: {message}"
        message = self.preproc_content(message)
        assert message is not None and self.check_content(message) == "listdict"
        for item in message:
            assert item["type"] in self.allowed_types, (
                f"Invalid input type: {item['type']}"
            )
        return self.generate_inner(message, dataset)

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset=None):
        if hasattr(self, "dump_image_func") and self.dump_image_func is not None:
            return self.dump_image_func(line)
        return line

    def use_custom_prompt(self, dataset):
        return False

    def build_prompt(self, line, dataset=None):
        raise NotImplementedError
