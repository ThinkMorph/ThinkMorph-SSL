"""Interleaved reasoning model (ThinkMorph-SSL checkpoints) for VLMEvalKit.

Loads a merged safetensors checkpoint and uses the repository's
InterleaveInferencer with action-token control for text + image + text reasoning.

Merge a training checkpoint first (run once per checkpoint):
  python scripts/merge_checkpoint.py <fsdp2_ckpt_dir> <merged_dir> <base_model_path> --checkpoint-type ema
"""

import json
import os
import random
import sys
import uuid

import numpy as np
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file as load_safetensors
from PIL import Image

from .base import BaseModel

# ---------------------------------------------------------------------------
# Import full model classes & inferencer from the repository's src/, so that
# evaluation runs exactly the model code used in training.
# Layout: <repo>/eval/VLMEvalKit/vlmeval/vlm/ThinkMorphInterleaved.py
# ---------------------------------------------------------------------------
_VLMEVAL_VLM_DIR = os.path.dirname(os.path.abspath(__file__))
_VLMEVALKIT_DIR = os.path.dirname(os.path.dirname(_VLMEVAL_VLM_DIR))
_REPO_ROOT = os.path.dirname(os.path.dirname(_VLMEVALKIT_DIR))
_BAGEL_SRC = os.environ.get("SSL_SRC", os.path.join(_REPO_ROOT, "src"))

if _BAGEL_SRC not in sys.path:
    sys.path.insert(0, _BAGEL_SRC)

from models.bagel import Bagel, BagelConfig  # noqa: E402
from models.bagel.qwen2_navit import Qwen2Config  # noqa: E402
from models.bagel.siglip_navit import SiglipVisionConfig  # noqa: E402
from models.bagel.autoencoder import load_ae  # noqa: E402
from models.bagel.data_utils import add_special_tokens  # noqa: E402
from models.bagel.inferencer import InterleaveInferencer  # noqa: E402
from models.bagel.transforms import ImageTransform  # noqa: E402
from transformers import Qwen2Tokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading (same pattern as ThinkMorph.py but with project's Bagel)
# ---------------------------------------------------------------------------

def _build_and_load_model(model_path, target_device="cuda:0"):
    """Build Bagel model and load weights from merged safetensors."""
    llm_cfg = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_cfg.qk_norm = True
    llm_cfg.tie_word_embeddings = False
    llm_cfg.layer_module = "Qwen2MoTDecoderLayer"

    vit_cfg = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_cfg.rope = False
    vit_cfg.num_hidden_layers -= 1

    vae_model, vae_cfg = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True, visual_und=True,
        llm_config=llm_cfg.to_dict(), vit_config=vit_cfg.to_dict(), vae_config=vae_cfg,
        vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
        latent_patch_size=2, max_latent_size=64,
    )

    with init_empty_weights():
        model = Bagel(config)

    # A merged training checkpoint has model.safetensors; the released weights have ema.safetensors.
    checkpoint = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(checkpoint):
        checkpoint = os.path.join(model_path, "ema.safetensors")
    sd = load_safetensors(checkpoint, device="cpu")
    model.load_state_dict(sd, assign=True, strict=False)
    del sd

    import gc; gc.collect()
    model = model.to(target_device).eval()
    return model, vae_model


class ThinkMorphInterleaved(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    DEFAULT_SYSTEM_PROMPT = (
        "Any image you need to produce — whether as a reasoning aid or as a "
        "final output — is emitted by writing <gen_image> immediately followed "
        "by an image-size declaration in the form [Image size: (height, width)], "
        "e.g. <gen_image>[Image size: (512, 512)]. Once an image is generated, "
        "it becomes part of your observable context, and you can continue "
        "reasoning or generating further images based on what you produced.\n\n"
        "For reasoning questions, wrap your thinking in <think> </think> and "
        "your final answer in <answer> </answer>. Inside <think>, you may "
        "freely interleave text reasoning and generated images across multiple "
        "rounds — think, generate an image, think about what you produced, "
        "generate another image if helpful, and so on. Use generated images "
        "only when they genuinely aid your reasoning; when text alone suffices, "
        "just reason in words.\n\n"
        "For tasks that directly ask for an image (such as generation or "
        "editing), produce the image without <think> or <answer> wrappers."
    )

    def __init__(self, model_path, save_dir="outputs_interleaved", **kwargs):
        """
        Args:
            model_path: Path to merged checkpoint dir (with model.safetensors or
                        ema.safetensors, llm_config.json, vit_config.json,
                        ae.safetensors, tokenizer).
        """
        assert model_path is not None
        self.model_path = model_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Distributed detection
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1

        # torchrun sets CUDA_VISIBLE_DEVICES per rank → each sees 1 GPU as cuda:0
        target_device = "cuda:0"

        print(f"[Rank {self.rank}] Loading model from {model_path}", flush=True)
        model, vae_model = _build_and_load_model(model_path, target_device=target_device)
        vae_model = vae_model.to(target_device).to(torch.bfloat16).eval()
        print(f"[Rank {self.rank}] Model loaded on {target_device}", flush=True)

        # Tokenizer (from merged dir – has <gen_image> token)
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        action_token = "<gen_image>"
        action_token_id = tokenizer.convert_tokens_to_ids(action_token)
        new_token_ids["image_action_token_id"] = action_token_id
        new_token_ids["register_action_token"] = True

        # Transforms
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # Inferencer
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform

        self.inferencer = InterleaveInferencer(
            model=model, vae_model=vae_model, tokenizer=tokenizer,
            vae_transform=vae_transform, vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        # Inference hyperparameters (aligned with training probe config)
        self.system_prompt = kwargs.get(
            "action_control_system_prompt", self.DEFAULT_SYSTEM_PROMPT
        )
        self.inference_hyper = dict(
            think=True,
            understanding_output=False,
            use_action_token_control=True,
            use_action_control_system_prompt=True,
            action_control_system_prompt=self.system_prompt,
            image_action_token=action_token,
            enable_size_tokens=True,
            enable_ref_image=False,
            do_sample=kwargs.get("do_sample", os.environ.get("EVAL_DO_SAMPLE", "false").lower() == "true"),
            text_temperature=kwargs.get("text_temperature", float(os.environ.get("EVAL_TEXT_TEMPERATURE", "0.3"))),
            cfg_text_scale=kwargs.get("cfg_text_scale", 3.0),
            cfg_img_scale=kwargs.get("cfg_img_scale", 1.5),
            cfg_interval=list(kwargs.get("cfg_interval", [0.4, 1.0])),
            cfg_renorm_type=kwargs.get("cfg_renorm_type", "global"),
            cfg_renorm_min=kwargs.get("cfg_renorm_min", 1.0),
            timestep_shift=kwargs.get("timestep_shift", 3.0),
            num_timesteps=kwargs.get("num_timesteps", int(os.environ.get("EVAL_NUM_TIMESTEPS", "50"))),
            max_turn_text_tokens=kwargs.get("max_turn_text_tokens", 1024),
            max_think_token_n=kwargs.get("max_think_token_n", int(os.environ.get("EVAL_MAX_THINK_TOKEN_N", "1024"))),
            image_shapes=tuple(kwargs.get("image_shapes", (512, 512))),
            feedback_generated_image_to_context=True,
            feedback_generated_image_as_vae_condition=True,
            feedback_generated_image_as_vit_condition=True,
            inline_image_latents=True,
        )

        # Reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def build_input(self, message):
        images = []
        text_parts = []
        image_counter = 1
        for m in message:
            if m["type"] == "image":
                val = m["value"]
                img = Image.open(val).convert("RGB") if isinstance(val, str) else val
                images.append(img)
                text_parts.append(f"<img><|image_{image_counter}|></img>")
                image_counter += 1
            elif m["type"] == "text":
                text_parts.append(m["value"])
        return images + [" ".join(text_parts)]

    def generate_inner(self, message, dataset=None):
        input_list = self.build_input(message)
        output_list = self.inferencer.interleave_inference(
            input_list, **self.inference_hyper
        )

        # Save reasoning trace
        sample_id = uuid.uuid4().hex[:8]
        trace_parts = []
        for idx, item in enumerate(output_list):
            if isinstance(item, str):
                trace_parts.append(item)
            elif isinstance(item, Image.Image):
                img_path = os.path.join(self.save_dir, f"reasoning_{sample_id}_{idx}.png")
                item.save(img_path)
                trace_parts.append(f"[Generated Image: {img_path}]")
            elif isinstance(item, dict):
                if "text" in item and item["text"]:
                    trace_parts.append(str(item["text"]))
                if "image" in item and isinstance(item["image"], Image.Image):
                    img_path = os.path.join(self.save_dir, f"reasoning_{sample_id}_{idx}.png")
                    item["image"].save(img_path)
                    trace_parts.append(f"[Generated Image: {img_path}]")

        full_text = "\n".join(trace_parts)
        trace_path = os.path.join(self.save_dir, f"trace_{sample_id}.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump({"sample_id": sample_id, "trace": trace_parts}, f, ensure_ascii=False, indent=2)

        return full_text
