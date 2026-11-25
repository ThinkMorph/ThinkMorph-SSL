import math
import os
import random
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image
from transformers import Qwen2Tokenizer, set_seed

from data.registry import register_processor
from models.bagel.data_utils import (
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
    len2weight,
    patchify,
    prepare_attention_mask_per_sample,
)
from models.bagel.transforms import ImageTransform

from .config import ProcessorConfig


@register_processor("bagel")
class BagelDataProcessor:
    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config
        extra_kwargs = getattr(self.config, "extra_kwargs", None) or {}
        self.vit_patch_size = int(extra_kwargs.get("vit_patch_size", 14))
        self.max_num_patch_per_side = int(
            extra_kwargs.get("max_num_patch_per_side", 70)
        )
        self.interpolate_pos = bool(extra_kwargs.get("interpolate_pos", False))
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate
        # Default latent * downsample = 2 * 8
        self.vae_image_downsample = int(extra_kwargs.get("vae_image_downsample", 16))
        self.max_latent_size = int(extra_kwargs.get("max_latent_size", 64))

        self.text_cond_dropout_prob = float(
            extra_kwargs.get("text_cond_dropout_prob", 0.0)
        )
        self.vit_cond_dropout_prob = float(
            extra_kwargs.get("vit_cond_dropout_prob", 0.0)
        )
        self.vae_cond_dropout_prob = float(
            extra_kwargs.get("vae_cond_dropout_prob", 0.0)
        )
        # Keep assistant-image ViT context so later text/image blocks can attend to
        # semantic features from previous assistant images.
        self.uniform_timestep_sampling = bool(
            extra_kwargs.get("uniform_timestep_sampling", False)
        )
        self.assistant_image_as_vit_condition = bool(
            extra_kwargs.get("assistant_image_as_vit_condition", True)
        )
        # Keep assistant-image clean VAE latents as explicit generation conditions
        # for later image blocks (teacher forcing in training).
        self.assistant_image_as_vae_condition = bool(
            extra_kwargs.get("assistant_image_as_vae_condition", True)
        )
        # When enabled, inject image dimension text (e.g., "(256, 256)") before
        # user images and after assistant action tokens so the model learns to
        # predict the target generation size.
        self.enable_size_tokens = bool(extra_kwargs.get("enable_size_tokens", False))
        # When False, the action token is tokenized as plain text (multi-token)
        # instead of being registered as a new special token.  This eliminates
        # the need for ce_loss_reweighting and EMA delay.
        self.register_action_token = bool(
            extra_kwargs.get("register_action_token", True)
        )
        # Action token used for explicit text-driven image triggering.
        self.image_action_token = str(
            extra_kwargs.get("image_action_token", "<|gen_image|>")
        )
        # Extra CE multiplier applied only when the target label is image_action_token.
        # Effective only when model config enables ce_loss_reweighting.
        self.action_token_ce_loss_weight = float(
            extra_kwargs.get("action_token_ce_loss_weight", 1.0)
        )
        if self.action_token_ce_loss_weight <= 0:
            raise ValueError(
                "action_token_ce_loss_weight must be > 0, got "
                f"{self.action_token_ce_loss_weight}."
            )
        self.supervise_image_action_token = bool(
            extra_kwargs.get("supervise_image_action_token", True)
        )
        # Optional paired closing marker that follows the image latents in
        # assistant generation. When register_action_end_token is True the
        # token is registered in the tokenizer and the processor emits it
        # right after <|vision_end|> in training data, giving the model an
        # explicit "image segment closed, continue with text" signal that is
        # not overloaded with user-image / final-turn semantics.
        # Default OFF: prior experiments and prior checkpoints behave
        # identically. Opt in by setting register_action_end_token: true.
        self.image_action_end_token = str(
            extra_kwargs.get("image_action_end_token", "<gen_image_end>")
        )
        self.register_action_end_token = bool(
            extra_kwargs.get("register_action_end_token", False)
        )
        self.supervise_image_action_end_token = bool(
            extra_kwargs.get("supervise_image_action_end_token", True)
        )
        # When True, action-token CE reweighting applies to the whole action
        # suffix: ``<|gen_image|>[Image size: (H, W)](ref: N)``.
        self.reweight_action_suffix_tokens = bool(
            extra_kwargs.get("reweight_action_suffix_tokens", True)
        )
        self.text_can_attend_vae_condition = bool(
            extra_kwargs.get("text_can_attend_vae_condition", True)
        )
        if (
            "text_can_attend_user_vae_condition" in extra_kwargs
            or "text_can_attend_assistant_vae_condition" in extra_kwargs
        ):
            logger.warning(
                "text_can_attend_user_vae_condition/text_can_attend_assistant_vae_condition "
                "are deprecated; use text_can_attend_vae_condition."
            )
        # Fine-grained control for image-generation target queries
        # (noise blocks) attending previous VAE-condition blocks.
        self.ref_target_can_attend_vae_condition = bool(
            extra_kwargs.get("ref_target_can_attend_vae_condition", False)
        )
        self.no_ref_target_can_attend_vae_condition = bool(
            extra_kwargs.get("no_ref_target_can_attend_vae_condition", True)
        )
        # When enabled, assistant images with a ref annotation use the ref
        # image's VAE latent as the flow-matching starting point instead of
        # random noise, and the model learns to predict ``(ref: N)`` tokens.
        self.enable_ref_image = bool(extra_kwargs.get("enable_ref_image", False))
        # When True, image latents are placed inline in the text stream without
        # EOS/BOS boundaries.  This produces a continuous sequence:
        #   [BOS text action(H,W)] [SOI latents EOI] [continuation EOS]
        # instead of the segmented:
        #   [BOS text action(H,W) EOS] [SOI latents EOI] [BOS continuation EOS]
        self.inline_image_latents = bool(extra_kwargs.get("inline_image_latents", True))

        self.user_image_as_vae_condition = extra_kwargs.get(
            "user_image_as_vae_condition", None
        )
        # When True, apply vae_cond_dropout_prob to user image VAE conditions
        # too (not just assistant images). This matches ThinkMorph's behavior
        # where input image VAE conditions are subject to dropout.
        self.user_vae_cond_dropout = bool(
            extra_kwargs.get("user_vae_cond_dropout", False)
        )
        self.set_random_seed(extra_kwargs.get("random_seed", 4396))

    def build(self):
        self.processor = self._build_processor()

    def set_random_seed(self, seed: int):
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0
        seed = seed * world_size + rank
        logger.info(
            f"Set random seed to {seed} for rank {rank} in world size {world_size}"
        )
        set_seed(seed)

    def _build_processor(self):
        extra_kwargs = getattr(self.config, "extra_kwargs", None) or {}
        processor = Qwen2Tokenizer.from_pretrained(self.config.processor_name)
        processor, self.new_token_ids, self.num_new_tokens = self.add_special_tokens(
            processor
        )
        self.vae_image_stride = int(extra_kwargs.get("vae_image_stride", 16))
        self.vae_max_image_size = int(extra_kwargs.get("vae_max_image_size", 1024))
        self.vae_min_image_size = int(extra_kwargs.get("vae_min_image_size", 512))
        self.vae_image_transform = ImageTransform(
            image_stride=self.vae_image_stride,
            max_image_size=self.vae_max_image_size,
            min_image_size=self.vae_min_image_size,
        )
        self.vit_image_stride = int(extra_kwargs.get("vit_image_stride", 14))
        self.vit_max_image_size = int(extra_kwargs.get("vit_max_image_size", 512))
        self.vit_min_image_size = int(extra_kwargs.get("vit_min_image_size", 378))
        self.vit_max_pixels = int(extra_kwargs.get("vit_max_pixels", 2_007_040))
        self.vit_image_transform = ImageTransform(
            image_stride=self.vit_image_stride,
            max_image_size=self.vit_max_image_size,
            min_image_size=self.vit_min_image_size,
            max_pixels=self.vit_max_pixels,
        )
        return processor

    def _get_image_size_text(self, image: Image.Image) -> str:
        """Return VAE-resized dimensions as a labeled text block.

        The same ``[Image size: (H, W)]`` label is used for both user input
        images and assistant-generated images so the model sees a consistent
        size representation regardless of which side produced the image.
        """
        resized = self.vae_image_transform.resize_transform(image.convert("RGB"))
        w, h = resized.size  # PIL .size returns (width, height)
        return f"[Image size: ({h}, {w})]"

    def save_pretrained(self, save_directory: str):
        if not hasattr(self, "processor"):
            raise ValueError(
                "Processor has not been built yet. Please call build() first."
            )
        # Build a clean processor for saving
        new_processor = self._build_processor()
        new_processor.save_pretrained(save_directory)

    def _resolve_ref_image_index(
        self,
        ref_image_mapping: Optional[List[Optional[int]]],
        image_index: int,
        max_valid_ref_index: int,
    ) -> Optional[int]:
        """Resolve a validated ref image index for the current image.

        Args:
            ref_image_mapping: Per-image mapping to unified ref indexes.
            image_index: Current image index in the unified image order.
            max_valid_ref_index: Largest allowed ref index (inclusive).
        """
        if not self.enable_ref_image or ref_image_mapping is None:
            return None
        if image_index < 0 or image_index >= len(ref_image_mapping):
            return None
        ref_idx = ref_image_mapping[image_index]
        if ref_idx is None:
            return None
        try:
            ref_idx = int(ref_idx)
        except (TypeError, ValueError):
            return None
        if ref_idx < 0 or ref_idx > max_valid_ref_index:
            return None
        return ref_idx

    def process(
        self,
        images: List[Image.Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        add_system_prompt=True,
        ref_image_mapping: Optional[List[Optional[int]]] = None,
        **kwargs,
    ):
        """
        A wrapper method to process single data
        """
        images = images or []
        image_index = 0
        # text = []
        # vae_images = []
        # vit_images = []
        sequence_status = self.set_sequence_status()
        # process_order = []
        curr = 0
        curr_rope_id = 0
        full_attn_modes = []
        split_lens = []
        split_types = []

        def _record_split(attn_modes, curr_split_len, split_type: str):
            full_attn_modes.extend(attn_modes)
            split_lens.append(curr_split_len)
            split_types.extend([split_type] * len(attn_modes))

        use_user_vae_cond = self.user_image_as_vae_condition
        if use_user_vae_cond is None:
            has_assistant_image = any(
                (m.get("role") == "assistant")
                and any((c.get("type") == "image") for c in (m.get("content") or []))
                for m in hf_messages
            )
            use_user_vae_cond = has_assistant_image

        prev_was_inline_image = False
        for message in hf_messages:
            role = message["role"]
            contents = message["content"]
            prev_was_inline_image = False  # reset per message
            for ci, content in enumerate(contents):
                ctype = content.get("type")
                if ctype == "text":
                    if role != "assistant" and self.text_cond_dropout_prob > 0:
                        if random.random() < self.text_cond_dropout_prob:
                            continue
                    curr_split_len = 0
                    curr_text = content["text"]
                    # Lookahead: if the next content is an assistant image,
                    # inline the action token at the end of this text block
                    # so the model learns to predict it during normal text
                    # generation (no separate [BOS action EOS] block needed).
                    action_suffix = None
                    action_suffix_text = None
                    next_is_assistant_image = False
                    if (
                        self.supervise_image_action_token
                        and role == "assistant"
                        and ci + 1 < len(contents)
                        and contents[ci + 1].get("type") == "image"
                    ):
                        next_is_assistant_image = True
                    if next_is_assistant_image:
                        suffix_parts = []
                        if self.register_action_token:
                            action_suffix = self.image_action_token_id
                        else:
                            # Plain-text mode: include the action token string
                            # as text instead of a single token ID.
                            suffix_parts.append(self.image_action_token)
                        if self.enable_size_tokens:
                            suffix_parts.append(
                                self._get_image_size_text(images[image_index])
                            )
                        if self.enable_ref_image and ref_image_mapping is not None:
                            ref_idx = self._resolve_ref_image_index(
                                ref_image_mapping=ref_image_mapping,
                                image_index=image_index,
                                max_valid_ref_index=image_index - 1,
                            )
                            if ref_idx is not None:
                                suffix_parts.append(f"(ref: {ref_idx})")
                        if suffix_parts:
                            action_suffix_text = "".join(suffix_parts)
                    # Inline image latents: omit EOS before image, omit BOS after image
                    inline_omit_bos = (
                        self.inline_image_latents
                        and role == "assistant"
                        and prev_was_inline_image
                    )
                    inline_omit_eos = (
                        self.inline_image_latents and next_is_assistant_image
                    )
                    inline_last_label = self.start_of_image if inline_omit_eos else None
                    attn_modes, sequence_status, curr, curr_split_len, curr_rope_id = (
                        self.process_text(
                            curr_text,
                            role,
                            sequence_status,
                            curr,
                            curr_split_len,
                            curr_rope_id,
                            action_suffix_id=action_suffix,
                            action_suffix_text=action_suffix_text,
                            omit_bos=inline_omit_bos,
                            omit_eos=inline_omit_eos,
                            last_label_id=inline_last_label,
                        )
                    )
                    _record_split(attn_modes, curr_split_len, "text")
                    if not next_is_assistant_image:
                        prev_was_inline_image = False
                    continue
                if ctype == "image":
                    curr_image_index = image_index
                    curr_image = images[curr_image_index]
                    image_index += 1
                    if role == "assistant":
                        current_ref_idx = self._resolve_ref_image_index(
                            ref_image_mapping=ref_image_mapping,
                            image_index=curr_image_index,
                            max_valid_ref_index=curr_image_index - 1,
                        )
                        has_ref = current_ref_idx is not None
                        if self.supervise_image_action_token and (
                            ci == 0 or contents[ci - 1].get("type") != "text"
                        ):
                            # No preceding text to attach action token to;
                            # create a standalone [BOS action_token EOS] block.
                            standalone_suffix_parts = []
                            standalone_action_id = (
                                self.image_action_token_id
                                if self.register_action_token
                                else None
                            )
                            if not self.register_action_token:
                                standalone_suffix_parts.append(self.image_action_token)
                            if self.enable_size_tokens:
                                standalone_suffix_parts.append(
                                    self._get_image_size_text(curr_image)
                                )
                            if self.enable_ref_image and ref_image_mapping is not None:
                                ref_idx = self._resolve_ref_image_index(
                                    ref_image_mapping=ref_image_mapping,
                                    image_index=curr_image_index,
                                    max_valid_ref_index=curr_image_index - 1,
                                )
                                if ref_idx is not None:
                                    standalone_suffix_parts.append(f"(ref: {ref_idx})")
                            standalone_size_text = (
                                "".join(standalone_suffix_parts)
                                if standalone_suffix_parts
                                else None
                            )
                            # Inline: omit BOS if previous was inline image,
                            # always omit EOS since image follows immediately.
                            sa_omit_bos = (
                                self.inline_image_latents and prev_was_inline_image
                            )
                            sa_omit_eos = self.inline_image_latents
                            sa_last_label = (
                                self.start_of_image
                                if self.inline_image_latents
                                else None
                            )
                            curr_split_len = 0
                            (
                                attn_modes,
                                sequence_status,
                                curr,
                                curr_split_len,
                                curr_rope_id,
                            ) = self.process_text(
                                "",
                                role,
                                sequence_status,
                                curr,
                                curr_split_len,
                                curr_rope_id,
                                action_suffix_id=standalone_action_id,
                                action_suffix_text=standalone_size_text,
                                omit_bos=sa_omit_bos,
                                omit_eos=sa_omit_eos,
                                last_label_id=sa_last_label,
                            )
                            _record_split(attn_modes, curr_split_len, "text")
                        curr_split_len = 0
                        (
                            attn_modes,
                            sequence_status,
                            curr,
                            curr_split_len,
                            curr_rope_id,
                        ) = self.process_vae_image(
                            curr_image,
                            role,
                            sequence_status,
                            curr_rope_id=curr_rope_id,
                            curr=curr,
                            curr_split_len=curr_split_len,
                            need_loss=True,
                        )
                        _record_split(
                            attn_modes,
                            curr_split_len,
                            "vae_target_ref" if has_ref else "vae_target_no_ref",
                        )
                        if (
                            has_ref
                            and current_ref_idx is not None
                            and current_ref_idx < len(images)
                        ):
                            target_vae_idx = (
                                len(sequence_status["vae_image_tensors"]) - 1
                            )
                            if target_vae_idx >= 0:
                                target_tensor = sequence_status["vae_image_tensors"][
                                    target_vae_idx
                                ]
                                target_h = int(target_tensor.shape[1])
                                target_w = int(target_tensor.shape[2])
                                ref_image = (
                                    images[current_ref_idx]
                                    .convert("RGB")
                                    .resize((target_w, target_h), Image.LANCZOS)
                                )
                                ref_tensor = (
                                    self.vae_image_transform.to_tensor_transform(
                                        ref_image
                                    )
                                )
                                ref_tensor = (
                                    self.vae_image_transform.normalize_transform(
                                        ref_tensor
                                    )
                                )
                                sequence_status["ref_vae_image_tensors"].append(
                                    ref_tensor
                                )
                                sequence_status["ref_vae_target_indices"].append(
                                    target_vae_idx
                                )

                        if self.assistant_image_as_vae_condition:
                            # Feed back the ground-truth assistant image as clean VAE
                            # condition so later image targets can condition on it.
                            if (
                                self.vae_cond_dropout_prob > 0
                                and random.random() < self.vae_cond_dropout_prob
                            ):
                                curr_rope_id += 1
                            else:
                                curr_split_len = 0
                                (
                                    attn_modes,
                                    sequence_status,
                                    curr,
                                    curr_split_len,
                                    curr_rope_id,
                                ) = self.process_vae_image(
                                    curr_image,
                                    role,
                                    sequence_status,
                                    curr_rope_id=curr_rope_id,
                                    curr=curr,
                                    curr_split_len=curr_split_len,
                                    need_loss=False,
                                )
                                _record_split(
                                    attn_modes,
                                    curr_split_len,
                                    "assistant_vae_condition",
                                )
                        if self.assistant_image_as_vit_condition:
                            # Append assistant ViT context after its VAE target so that
                            # subsequent text/image segments can use it, while avoiding
                            # same-image leakage into the current VAE target block.
                            if (
                                self.vit_cond_dropout_prob > 0
                                and random.random() < self.vit_cond_dropout_prob
                            ):
                                curr_rope_id += 1
                            else:
                                curr_split_len = 0
                                (
                                    attn_modes,
                                    sequence_status,
                                    curr,
                                    curr_split_len,
                                    curr_rope_id,
                                ) = self.process_vit_image(
                                    curr_image,
                                    role,
                                    sequence_status,
                                    curr_rope_id=curr_rope_id,
                                    curr=curr,
                                    curr_split_len=curr_split_len,
                                )
                                _record_split(attn_modes, curr_split_len, "vit")
                        # Optional paired closing marker, emitted AFTER all
                        # image-related blocks (gen target + VAE/ViT cond) so
                        # it sits adjacent to the next text content rather
                        # than mid-way through the image emits. Gated on the
                        # opt-in flag; default off keeps behavior unchanged.
                        if self.register_action_end_token:
                            curr_split_len = 0
                            (
                                attn_modes,
                                sequence_status,
                                curr,
                                curr_split_len,
                                curr_rope_id,
                            ) = self._emit_image_action_end_token(
                                sequence_status,
                                curr,
                                curr_split_len,
                                curr_rope_id,
                            )
                            _record_split(
                                attn_modes, curr_split_len, "action_end_token"
                            )
                        if self.inline_image_latents:
                            prev_was_inline_image = True
                        continue
                    if role == "user":
                        if self.enable_size_tokens:
                            size_text = self._get_image_size_text(curr_image)
                            curr_split_len = 0
                            (
                                attn_modes,
                                sequence_status,
                                curr,
                                curr_split_len,
                                curr_rope_id,
                            ) = self.process_text(
                                size_text,
                                role,
                                sequence_status,
                                curr,
                                curr_split_len,
                                curr_rope_id,
                            )
                            _record_split(attn_modes, curr_split_len, "text")
                        if use_user_vae_cond:
                            # Optionally apply VAE condition dropout to user
                            # images (matches ThinkMorph's enable_cfg=1 behavior).
                            if (
                                self.user_vae_cond_dropout
                                and self.vae_cond_dropout_prob > 0
                                and random.random() < self.vae_cond_dropout_prob
                            ):
                                curr_rope_id += 1
                            else:
                                curr_split_len = 0
                                (
                                    attn_modes,
                                    sequence_status,
                                    curr,
                                    curr_split_len,
                                    curr_rope_id,
                                ) = self.process_vae_image(
                                    curr_image,
                                    role,
                                    sequence_status,
                                    curr_rope_id=curr_rope_id,
                                    curr=curr,
                                    curr_split_len=curr_split_len,
                                    need_loss=False,
                                )
                                _record_split(
                                    attn_modes, curr_split_len, "user_vae_condition"
                                )
                        curr_split_len = 0
                        (
                            attn_modes,
                            sequence_status,
                            curr,
                            curr_split_len,
                            curr_rope_id,
                        ) = self.process_vit_image(
                            curr_image,
                            role,
                            sequence_status,
                            curr_rope_id=curr_rope_id,
                            curr=curr,
                            curr_split_len=curr_split_len,
                        )
                        _record_split(attn_modes, curr_split_len, "vit")
                        continue
                continue

        sequence_status["attn_modes"] = full_attn_modes
        sequence_status["curr"] = curr
        sequence_status["sample_lens"].append(sum(split_lens))
        sequence_status["nested_attention_masks"].append(
            prepare_attention_mask_per_sample(
                split_lens,
                full_attn_modes,
                split_types=split_types,
                text_can_attend_vae_condition=self.text_can_attend_vae_condition,
                ref_target_can_attend_vae_condition=self.ref_target_can_attend_vae_condition,
                no_ref_target_can_attend_vae_condition=self.no_ref_target_can_attend_vae_condition,
            )
        )
        data = self.to_tensor(sequence_status)
        # Fake input ids for packing
        data["input_ids"] = torch.zeros(data["sequence_length"], dtype=torch.long)
        data["attention_mask"] = torch.ones(data["sequence_length"], dtype=torch.long)
        data["curr"] = curr
        # data["split_lens"] = split_lens

        return data

    def process_text(
        self,
        text: str,
        role: str,
        sequence_status: dict,
        curr: int,
        curr_split_len: int,
        curr_rope_id: int,
        action_suffix_id: int | None = None,
        action_suffix_text: str | None = None,
        omit_bos: bool = False,
        omit_eos: bool = False,
        last_label_id: int | None = None,
    ):
        attn_modes = []
        base_text_ids = self.processor.encode(text)
        action_suffix_token_ids = []
        if action_suffix_id is not None:
            action_suffix_token_ids.append(action_suffix_id)
        if action_suffix_text is not None:
            action_suffix_token_ids.extend(self.processor.encode(action_suffix_text))
        text_ids = base_text_ids + action_suffix_token_ids

        if omit_bos:
            shifted_text_ids = text_ids
        else:
            shifted_text_ids = [self.bos_token_id] + text_ids

        sequence_status["packed_text_ids"].extend(shifted_text_ids)
        sequence_status["packed_text_indexes"].extend(
            range(curr, curr + len(shifted_text_ids))
        )
        if role == "assistant":
            if omit_bos:
                # No BOS: labels are shifted within text_ids only.
                # text_ids[0] predicts text_ids[1], ..., text_ids[-1] predicts EOS.
                trailing_id = (
                    last_label_id if last_label_id is not None else self.eos_token_id
                )
                label_ids = text_ids[1:] + [trailing_id]
            else:
                trailing_id = (
                    last_label_id if last_label_id is not None else self.eos_token_id
                )
                label_ids = text_ids + [trailing_id]
            base_weight = len2weight(len(shifted_text_ids))
            ce_weights = [base_weight] * len(label_ids)
            if self.action_token_ce_loss_weight != 1.0:
                if (
                    self.reweight_action_suffix_tokens
                    and len(action_suffix_token_ids) > 0
                ):
                    # Compute suffix start relative to label_ids (not text_ids)
                    if omit_bos:
                        suffix_start = len(base_text_ids) - 1
                    else:
                        suffix_start = len(base_text_ids)
                    suffix_end = suffix_start + len(action_suffix_token_ids)
                    for i in range(
                        max(0, suffix_start), min(suffix_end, len(ce_weights))
                    ):
                        ce_weights[i] *= self.action_token_ce_loss_weight
                else:
                    ce_weights = [
                        w * self.action_token_ce_loss_weight
                        if tid == self.image_action_token_id
                        else w
                        for w, tid in zip(ce_weights, label_ids)
                    ]
            # Keep the trailing label (SOI) even when EOS is omitted.
            # Although inference stops decoding at ")" and never emits SOI,
            # the supervision on ")" → SOI is essential: without it the
            # hidden state at ")" is unconstrained, polluting the KV cache
            # and causing garbage tokens (e.g. "000") after size tokens.
            sequence_status["ce_loss_indexes"].extend(
                range(curr, curr + len(label_ids))
            )
            sequence_status["ce_loss_weights"].extend(ce_weights)
            sequence_status["packed_label_ids"].extend(label_ids)
        curr += len(shifted_text_ids)
        curr_split_len += len(shifted_text_ids)

        if not omit_eos:
            # add a <|im_end|> token
            sequence_status["packed_text_ids"].append(self.eos_token_id)
            sequence_status["packed_text_indexes"].append(curr)
            curr += 1
            curr_split_len += 1

        # update sequence status
        attn_modes.append("causal")
        sequence_status["packed_position_ids"].extend(
            range(curr_rope_id, curr_rope_id + curr_split_len)
        )
        curr_rope_id += curr_split_len
        return attn_modes, sequence_status, curr, curr_split_len, curr_rope_id

    def process_vae_image(
        self,
        # image_tensor: torch.Tensor,
        image: Image.Image,
        role: str,
        sequence_status: dict,
        curr_rope_id: int,
        curr: int,
        curr_split_len: int,
        need_loss: bool = True,
    ):
        image_tensor = self.vae_image_transform(image.convert("RGB"))
        attn_modes = []

        # Record generated image size (post-VAE-resize). Only count true
        # generation targets: role=assistant with loss enabled. Skip VAE-cond
        # passes (need_loss=False) and user VAE-cond (role=user).
        if role == "assistant" and need_loss:
            sequence_status["num_gen_images"] += 1
            sequence_status["gen_image_h"].append(int(image_tensor.size(1)))
            sequence_status["gen_image_w"].append(int(image_tensor.size(2)))

        # add a <|startofimage|> token
        sequence_status["packed_text_ids"].append(self.start_of_image)
        sequence_status["packed_text_indexes"].append(curr)
        curr += 1
        curr_split_len += 1

        # preprocess image
        sequence_status["vae_image_tensors"].append(image_tensor)
        sequence_status["packed_latent_position_ids"].append(
            self.get_flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.vae_image_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
        )
        H, W = image_tensor.shape[1:]
        h = H // self.vae_image_downsample
        w = W // self.vae_image_downsample
        sequence_status["vae_latent_shapes"].append((h, w))

        num_img_tokens = w * h
        sequence_status["packed_vae_token_indexes"].extend(
            range(curr, curr + num_img_tokens)
        )
        if need_loss:
            # Supervised image generation/editing target (align with original: loss=1).
            sequence_status["mse_loss_indexes"].extend(
                range(curr, curr + num_img_tokens)
            )
            _fixed_ts = os.environ.get("FIXED_TIMESTEP")
            if _fixed_ts is not None:
                timestep = float(_fixed_ts)
            elif self.uniform_timestep_sampling:
                t = random.uniform(1e-5, 1 - 1e-5)
                timestep = math.log(t / (1 - t))  # logit(t), sigmoid recovers uniform t
            else:
                timestep = np.random.randn()
        else:
            # VAE-condition image (align with original: loss=0, timestep=-inf -> sigmoid=0 -> clean latent condition).
            timestep = float("-inf")

        sequence_status["packed_timesteps"].extend([timestep] * num_img_tokens)
        curr += num_img_tokens
        curr_split_len += num_img_tokens

        # add a <|endofimage|> token
        sequence_status["packed_text_ids"].append(self.end_of_image)
        sequence_status["packed_text_indexes"].append(curr)
        curr += 1
        curr_split_len += 1

        sequence_status["packed_position_ids"].extend(
            [curr_rope_id] * (num_img_tokens + 2)
        )
        if need_loss:
            attn_modes.append("noise")
        else:
            # VAE-condition image (loss=0) uses full attention.
            # This allows VAE_cond and VAE_target (same image, different representations)
            # to attend to each other, strengthening semantic alignment.
            attn_modes.append("full")
            curr_rope_id += 1

        return attn_modes, sequence_status, curr, curr_split_len, curr_rope_id

    def process_vit_image(
        self,
        image: Image.Image,
        role: str,
        sequence_status: dict,
        curr_rope_id: int,
        curr: int,
        curr_split_len: int,
    ):
        attn_modes = []
        image_tensor = self.vit_image_transform(image.convert("RGB"))

        # Record user input image size (post-ViT-resize). Assistant-role ViT
        # calls are for inline image latents, not user inputs, so skip those.
        if role == "user":
            sequence_status["num_user_images"] += 1
            sequence_status["user_image_h"].append(int(image_tensor.size(1)))
            sequence_status["user_image_w"].append(int(image_tensor.size(2)))

        # add a <|vision_start|> token
        sequence_status["packed_text_ids"].append(self.start_of_image)
        sequence_status["packed_text_indexes"].append(curr)
        curr += 1
        curr_split_len += 1

        # add the image tensor
        vit_tokens = patchify(image_tensor, self.vit_patch_size)
        num_img_tokens = vit_tokens.shape[0]
        sequence_status["packed_vit_token_indexes"].extend(
            range(curr, curr + num_img_tokens)
        )
        curr += num_img_tokens
        curr_split_len += num_img_tokens

        sequence_status["packed_vit_tokens"].append(vit_tokens)
        sequence_status["vit_token_seqlens"].append(num_img_tokens)
        sequence_status["packed_vit_position_ids"].append(
            self.get_flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.vit_patch_size,
                max_num_patches_per_side=self.max_num_patch_per_side,
            )
        )

        # add a <|endofimage|> token
        sequence_status["packed_text_ids"].append(self.end_of_image)
        sequence_status["packed_text_indexes"].append(curr)

        # if item['special_token_loss'] == 1: # <|endofimage|> may have loss
        #     sequence_status['ce_loss_indexes'].append(curr)
        #     sequence_status['ce_loss_weights'].append(1.0)
        #     sequence_status['packed_label_ids'].append(item['special_token_label'])

        curr += 1
        curr_split_len += 1

        # update sequence status
        attn_modes.append("full")
        sequence_status["packed_position_ids"].extend([curr_rope_id] * curr_split_len)
        curr_rope_id += 1

        return attn_modes, sequence_status, curr, curr_split_len, curr_rope_id

    def _emit_image_action_end_token(
        self,
        sequence_status: dict,
        curr: int,
        curr_split_len: int,
        curr_rope_id: int,
    ):
        """Emit a single <gen_image_end> closing token for an assistant image.

        Called by the message loop after all image-related emits for this
        content (gen target + optional VAE/ViT conditioning) have completed,
        so the close token sits between the last image segment and the next
        text content. Supervises the immediately preceding position to predict
        the close token id.

        Returns the standard (attn_modes, sequence_status, curr,
        curr_split_len, curr_rope_id) tuple so the caller can _record_split.
        """
        attn_modes = []
        if self.supervise_image_action_end_token:
            # The token at position curr-1 (last token of the prior image
            # segment) is supervised to predict the close token id at curr.
            sequence_status["ce_loss_indexes"].append(curr - 1)
            sequence_status["ce_loss_weights"].append(1.0)
            sequence_status["packed_label_ids"].append(
                self.image_action_end_token_id
            )
        sequence_status["packed_text_ids"].append(self.image_action_end_token_id)
        sequence_status["packed_text_indexes"].append(curr)
        sequence_status["packed_position_ids"].append(curr_rope_id)
        curr += 1
        curr_split_len += 1
        attn_modes.append("causal")
        curr_rope_id += 1
        return attn_modes, sequence_status, curr, curr_split_len, curr_rope_id

    def set_sequence_status(self):
        sequence_status = dict(
            curr=0,
            sample_lens=list(),
            packed_position_ids=list(),
            nested_attention_masks=list(),
            split_lens=list(),
            attn_modes=list(),
            packed_text_ids=list(),
            packed_text_indexes=list(),
            packed_label_ids=list(),
            ce_loss_indexes=list(),
            ce_loss_weights=list(),
            vae_image_tensors=list(),
            packed_latent_position_ids=list(),
            vae_latent_shapes=list(),
            packed_vae_token_indexes=list(),
            packed_timesteps=list(),
            mse_loss_indexes=list(),
            packed_vit_tokens=list(),
            vit_token_seqlens=list(),
            packed_vit_position_ids=list(),
            packed_vit_token_indexes=list(),
            ref_vae_image_tensors=list(),
            ref_vae_target_indices=list(),
            # Per-sample data stats for wandb logging. Populated by
            # process_vit_image (user inputs) and process_vae_image
            # (assistant gen targets). Post-resize (tokenizable) H/W.
            num_user_images=0,
            num_gen_images=0,
            user_image_h=list(),
            user_image_w=list(),
            gen_image_h=list(),
            gen_image_w=list(),
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status["sample_lens"]),
            sample_lens=sequence_status["sample_lens"],
            packed_text_ids=torch.tensor(sequence_status["packed_text_ids"]),
            packed_text_indexes=torch.tensor(sequence_status["packed_text_indexes"]),
            packed_position_ids=torch.tensor(sequence_status["packed_position_ids"]),
        )

        data["nested_attention_masks"] = sequence_status["nested_attention_masks"]

        # Per-sample data stats for wandb logging.
        data["num_user_images"] = int(sequence_status["num_user_images"])
        data["num_gen_images"] = int(sequence_status["num_gen_images"])
        data["user_image_h"] = list(sequence_status["user_image_h"])
        data["user_image_w"] = list(sequence_status["user_image_w"])
        data["gen_image_h"] = list(sequence_status["gen_image_h"])
        data["gen_image_w"] = list(sequence_status["gen_image_w"])

        # if the model has a convent vae (e.g., as visual tokenizer)
        if len(sequence_status["vae_image_tensors"]) > 0:
            image_tensors = sequence_status.pop("vae_image_tensors")
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[
                    i, :, : image_tensor.shape[1], : image_tensor.shape[2]
                ] = image_tensor

            data["padded_images"] = padded_images
            data["patchified_vae_latent_shapes"] = sequence_status["vae_latent_shapes"]
            data["packed_latent_position_ids"] = torch.cat(
                sequence_status["packed_latent_position_ids"], dim=0
            )
            data["packed_vae_token_indexes"] = torch.tensor(
                sequence_status["packed_vae_token_indexes"]
            )
            if len(sequence_status["ref_vae_image_tensors"]) > 0:
                ref_image_tensors = sequence_status["ref_vae_image_tensors"]
                ref_sizes = [item.shape for item in ref_image_tensors]
                max_ref_size = [max(item) for item in list(zip(*ref_sizes))]
                ref_padded_images = torch.zeros(
                    size=(len(ref_image_tensors), *max_ref_size),
                    dtype=ref_image_tensors[0].dtype,
                )
                for i, image_tensor in enumerate(ref_image_tensors):
                    ref_padded_images[
                        i, :, : image_tensor.shape[1], : image_tensor.shape[2]
                    ] = image_tensor
                data["ref_padded_images"] = ref_padded_images
                data["ref_vae_target_indices"] = torch.tensor(
                    sequence_status["ref_vae_target_indices"], dtype=torch.long
                )

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status["packed_vit_tokens"]) > 0:
            data["packed_vit_tokens"] = torch.cat(
                sequence_status["packed_vit_tokens"], dim=0
            )
            data["packed_vit_position_ids"] = torch.cat(
                sequence_status["packed_vit_position_ids"], dim=0
            )
            data["packed_vit_token_indexes"] = torch.tensor(
                sequence_status["packed_vit_token_indexes"]
            )
            data["vit_token_seqlens"] = torch.tensor(
                sequence_status["vit_token_seqlens"]
            )

        # if the model is required to perform visual generation
        if len(sequence_status["packed_timesteps"]) > 0:
            data["packed_timesteps"] = torch.tensor(sequence_status["packed_timesteps"])
            data["mse_loss_indexes"] = torch.tensor(sequence_status["mse_loss_indexes"])

        # if the model is required to perform text generation
        if len(sequence_status["packed_label_ids"]) > 0:
            data["packed_label_ids"] = torch.tensor(sequence_status["packed_label_ids"])
            data["ce_loss_indexes"] = torch.tensor(sequence_status["ce_loss_indexes"])
            data["ce_loss_weights"] = torch.tensor(sequence_status["ce_loss_weights"])

        return data

    @property
    def image_token_id(self):
        image_token = getattr(self.processor, "image_token", None)
        if image_token is None:
            return None
        else:
            return self.processor.tokenizer.convert_tokens_to_ids(
                self.processor.image_token
            )

    @property
    def video_token_id(self):
        video_token = getattr(self.processor, "video_token", None)
        if video_token is None:
            return None
        else:
            return self.processor.tokenizer.convert_tokens_to_ids(
                self.processor.video_token
            )

    @property
    def tokenizer(self):
        return self.tokenizer

    def add_special_tokens(self, tokenizer):
        all_special_tokens = []
        for k, v in tokenizer.special_tokens_map.items():
            if isinstance(v, str):
                all_special_tokens.append(v)
            elif isinstance(v, list):
                all_special_tokens += v

        new_tokens = []

        if "<|im_start|>" not in all_special_tokens:
            new_tokens.append("<|im_start|>")

        if "<|im_end|>" not in all_special_tokens:
            new_tokens.append("<|im_end|>")

        if "<|vision_start|>" not in all_special_tokens:
            new_tokens.append("<|vision_start|>")

        if "<|vision_end|>" not in all_special_tokens:
            new_tokens.append("<|vision_end|>")
        if (
            self.register_action_token
            and self.image_action_token not in all_special_tokens
        ):
            new_tokens.append(self.image_action_token)

        if (
            self.register_action_end_token
            and self.image_action_end_token not in all_special_tokens
        ):
            new_tokens.append(self.image_action_end_token)

        num_new_tokens = tokenizer.add_tokens(new_tokens)
        bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.bos_token_id = bos_token_id
        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_token_id = eos_token_id
        start_of_image = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        end_of_image = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        reserved_ids = {
            bos_token_id,
            eos_token_id,
            start_of_image,
            end_of_image,
        }
        if self.register_action_token:
            image_action_token_id = tokenizer.convert_tokens_to_ids(
                self.image_action_token
            )
            if image_action_token_id in reserved_ids:
                raise ValueError(
                    "image_action_token must be distinct from "
                    "<|im_start|>, <|im_end|>, <|vision_start|>, <|vision_end|>. "
                    f"Got {self.image_action_token!r} (id={image_action_token_id})."
                )
        else:
            # Plain-text mode: the action token is multi-token text, so there
            # is no single dedicated token id.  Use -1 as a sentinel to avoid
            # false matches in per-token-id CE reweighting.
            image_action_token_id = -1

        if self.register_action_end_token:
            image_action_end_token_id = tokenizer.convert_tokens_to_ids(
                self.image_action_end_token
            )
            extra_reserved = reserved_ids | (
                {image_action_token_id} if image_action_token_id != -1 else set()
            )
            if image_action_end_token_id in extra_reserved:
                raise ValueError(
                    "image_action_end_token must be distinct from "
                    "<|im_start|>, <|im_end|>, <|vision_start|>, <|vision_end|>, "
                    "and image_action_token. "
                    f"Got {self.image_action_end_token!r} "
                    f"(id={image_action_end_token_id})."
                )
        else:
            image_action_end_token_id = -1

        self.start_of_image = start_of_image
        self.end_of_image = end_of_image
        self.image_action_token_id = image_action_token_id
        self.image_action_end_token_id = image_action_end_token_id

        new_token_ids = dict(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            start_of_image=start_of_image,
            end_of_image=end_of_image,
            image_action_token_id=image_action_token_id,
            register_action_token=self.register_action_token,
            image_action_end_token_id=image_action_end_token_id,
            register_action_end_token=self.register_action_end_token,
        )

        return tokenizer, new_token_ids, num_new_tokens
