# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from .data_utils import pil_img2rgb
from .qwen2_navit import NaiveCache

logger = logging.getLogger(__name__)

VLM_THINK_SYSTEM_PROMPT = """You should first think about the reasoning process in the mind and then provide the user with the answer.
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"""

GEN_THINK_SYSTEM_PROMPT = """
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
"""

def build_action_control_system_prompt(image_action_token: str = "<|gen_image|>") -> str:
    token = image_action_token or "<|gen_image|>"
    return (
        "You should first think about the reasoning and planning process in the "
        "mind and then provide the user with the answer or generate the image.\n"
        "The thinking process is enclosed within <think> </think> tags.\n"
        f"If you need to visualize something to help your reasoning, output "
        f"{token} immediately followed by an image-size declaration in the form "
        f"[Image size: (height, width)], e.g. {token}[Image size: (512, 256)]. "
        f"An image will be generated and you can continue thinking based on the "
        f"result.\n"
        "You may generate multiple images during your reasoning process.\n"
        "Provide your final answer wrapped in <answer> </answer> tags."
    )


ACTION_CONTROL_SYSTEM_PROMPT = build_action_control_system_prompt()


class InterleaveInferencer:
    def __init__(
        self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids,
        cfg_parallel_workers=None,
    ):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        # Populated after every interleave_inference call for external logging.
        self.last_interleave_diagnostics: Dict[str, Any] = {}

        # Multi-GPU CFG parallel: {"base": {"model": ..., "device": ...}, "cfg_text": ..., "cfg_img": ...}
        self.cfg_parallel_workers = cfg_parallel_workers or {}
        self.cfg_parallel_executor = None
        if len(self.cfg_parallel_workers) > 1:
            from concurrent.futures import ThreadPoolExecutor
            self.cfg_parallel_executor = ThreadPoolExecutor(
                max_workers=len(self.cfg_parallel_workers),
                thread_name_prefix="cfg_parallel",
            )

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _to_model_device(self, obj):
        device = self._model_device()
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: self._to_model_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_model_device(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._to_model_device(v) for v in obj)
        return obj

    def _to_pil_image(self, image: Union[Image.Image, torch.Tensor]) -> Image.Image:
        if isinstance(image, Image.Image):
            return pil_img2rgb(image)

        if torch.is_tensor(image):
            image = image.detach().float().cpu()
            if image.dim() != 3:
                raise ValueError(
                    f"Expected image tensor with 3 dims (C,H,W) or (H,W,C), got {tuple(image.shape)}"
                )
            if image.shape[0] in (1, 3):
                image = image.permute(1, 2, 0)
            elif image.shape[-1] not in (1, 3):
                raise ValueError(
                    f"Unsupported image tensor shape for conversion: {tuple(image.shape)}"
                )
            image = (image.clamp(0, 1) * 255).to(torch.uint8).numpy()
            if image.shape[-1] == 1:
                image = image[..., 0]
            return pil_img2rgb(Image.fromarray(image))

        raise ValueError(f"Unsupported image type for conversion: {type(image)}")

    def _resolve_image_action_token_id(self, image_action_token: Optional[str]) -> int:
        reserved_ids = {
            int(self.new_token_ids.get("bos_token_id")),
            int(self.new_token_ids.get("eos_token_id")),
            int(self.new_token_ids.get("start_of_image")),
            int(self.new_token_ids.get("end_of_image")),
        }

        token_id = self.new_token_ids.get("image_action_token_id", None)
        if token_id is not None:
            if image_action_token:
                user_token_id = self.tokenizer.convert_tokens_to_ids(image_action_token)
                if user_token_id != int(token_id):
                    raise ValueError(
                        "image_action_token does not match tokenizer id registered in new_token_ids."
                    )
            if int(token_id) in reserved_ids:
                raise ValueError(
                    "Configured image_action_token_id collides with reserved boundary tokens. "
                    "Please use a dedicated token such as '<|gen_image|>'."
                )
            return int(token_id)

        token = image_action_token or "<|gen_image|>"
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise ValueError(
                f"Unable to resolve image action token id for token: {token}"
            )
        if int(token_id) in reserved_ids:
            raise ValueError(
                "image_action_token resolves to a reserved boundary token. "
                "Please use a dedicated token such as '<|gen_image|>'."
            )
        resolved_token = self.tokenizer.convert_ids_to_tokens(token_id)
        if resolved_token != token:
            raise ValueError(
                f"Token {token} is not registered as a standalone token; resolved to {resolved_token}."
            )
        return int(token_id)

    def _parse_size_text(self, text: str) -> Optional[Tuple[int, int]]:
        """Parse ``(H, W)`` from decoded text, returning an ``(H, W)`` tuple."""
        match = re.search(r"\((\d+),\s*(\d+)\)", text)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None

    def _clamp_image_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        """Clamp predicted image size to the model's supported latent position range.

        Prevents out-of-bounds indexing in ``latent_pos_embed`` when the model
        predicts sizes larger than ``max_latent_size * latent_downsample``.
        """
        max_dim = int(self.model.max_latent_size * self.model.latent_downsample)
        h, w = size
        if h > max_dim or w > max_dim:
            logger.warning(
                "Predicted image size (%d, %d) exceeds max supported %d; clamping.",
                h, w, max_dim,
            )
            h = min(h, max_dim)
            w = min(w, max_dim)
        # Ensure dimensions are multiples of latent_downsample
        ds = int(self.model.latent_downsample)
        h = max(ds, (h // ds) * ds)
        w = max(ds, (w // ds) * ds)
        return (h, w)

    def _decode_size_tokens(
        self,
        decode_state: Dict[str, Any],
        packed_query: torch.Tensor,
        max_tokens: int = 32,
        text_can_attend_vae_condition: bool = True,
    ) -> Tuple[Optional[Tuple[int, int]], torch.Tensor]:
        """Autoregressively decode ``[Image size: (H, W)]`` after the action token.

        Returns:
            parsed_size: Parsed ``(H, W)`` tuple or *None* on failure.
            packed_query: Last packed query state after decoding size tokens.
        """
        eos_token_id = int(self.new_token_ids["eos_token_id"])
        size_token_ids: List[int] = []
        for _ in range(max_tokens):
            next_token = self._sample_next_token_id(
                packed_query, do_sample=False, temperature=1.0
            )
            if next_token == eos_token_id:
                break
            size_token_ids.append(next_token)
            packed_query = self._forward_decode_text_token(
                decode_state,
                next_token,
                text_can_attend_vae_condition=text_can_attend_vae_condition,
            )
            decoded = self.tokenizer.decode(size_token_ids)
            # Stop once the closing bracket of ``[Image size: (H, W)]`` is seen.
            if "]" in decoded:
                break
        if not size_token_ids:
            return None, packed_query
        return self._parse_size_text(
            self.tokenizer.decode(size_token_ids)
        ), packed_query

    def _decode_ref_tokens(
        self,
        decode_state: Dict[str, Any],
        packed_query: torch.Tensor,
        max_tokens: int = 10,
        text_can_attend_vae_condition: bool = True,
    ) -> Tuple[Optional[int], torch.Tensor]:
        """Autoregressively decode ``(ref: N)`` tokens after action/size tokens."""
        eos_token_id = int(self.new_token_ids["eos_token_id"])
        ref_token_ids: List[int] = []
        for _ in range(max_tokens):
            next_token = self._sample_next_token_id(
                packed_query, do_sample=False, temperature=1.0
            )
            if next_token == eos_token_id:
                break
            ref_token_ids.append(next_token)
            packed_query = self._forward_decode_text_token(
                decode_state,
                next_token,
                text_can_attend_vae_condition=text_can_attend_vae_condition,
            )
            decoded = self.tokenizer.decode(ref_token_ids)
            if ")" in decoded:
                break
        if not ref_token_ids:
            return None, packed_query
        match = re.search(r"\(ref:\s*(\d+)\)", self.tokenizer.decode(ref_token_ids))
        if match is None:
            return None, packed_query
        return int(match.group(1)), packed_query

    def _init_single_decode_state(self, gen_context: Dict[str, Any]) -> Dict[str, Any]:
        if len(gen_context["kv_lens"]) != 1 or len(gen_context["ropes"]) != 1:
            raise ValueError("InterleaveInferencer currently supports batch size = 1.")
        self._ensure_context_metadata(gen_context)
        return {
            "past_key_values": gen_context["past_key_values"],
            "kv_len": int(gen_context["kv_lens"][0]),
            "rope": int(gen_context["ropes"][0]),
            "vae_condition_ranges": list(gen_context["vae_condition_ranges"][0]),
        }

    def _sync_decode_state_to_context(
        self, decode_state: Dict[str, Any], gen_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        self._ensure_context_metadata(gen_context)
        gen_context["past_key_values"] = decode_state["past_key_values"]
        gen_context["kv_lens"] = [int(decode_state["kv_len"])]
        gen_context["ropes"] = [int(decode_state["rope"])]
        return gen_context

    def _ensure_context_metadata(self, gen_context: Dict[str, Any]) -> None:
        kv_lens = gen_context.get("kv_lens", [])
        expected = len(kv_lens)
        ranges = gen_context.get("vae_condition_ranges")
        if not isinstance(ranges, list) or len(ranges) != expected:
            gen_context["vae_condition_ranges"] = [[] for _ in range(expected)]
            return
        normalized_ranges: List[List[Tuple[int, int]]] = []
        for sample_ranges in ranges:
            if not isinstance(sample_ranges, list):
                normalized_ranges.append([])
                continue
            current: List[Tuple[int, int]] = []
            for item in sample_ranges:
                if (
                    isinstance(item, (tuple, list))
                    and len(item) == 2
                    and isinstance(item[0], int)
                    and isinstance(item[1], int)
                ):
                    start = max(int(item[0]), 0)
                    end = max(int(item[1]), start)
                    current.append((start, end))
            normalized_ranges.append(current)
        gen_context["vae_condition_ranges"] = normalized_ranges

    def _build_allowed_kv_indexes(
        self,
        kv_len: int,
        vae_condition_ranges: List[Tuple[int, int]],
        allow_vae_condition: bool,
        device: torch.device,
    ) -> torch.Tensor:
        if allow_vae_condition or kv_len <= 0 or len(vae_condition_ranges) == 0:
            return torch.arange(kv_len, dtype=torch.long, device=device)
        keep_mask = torch.ones(kv_len, dtype=torch.bool, device=device)
        for start, end in vae_condition_ranges:
            start = max(0, min(int(start), kv_len))
            end = max(start, min(int(end), kv_len))
            if end > start:
                keep_mask[start:end] = False
        return torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)

    def _filter_cache_with_indexes(
        self, past_key_values: NaiveCache, keep_indexes: torch.Tensor
    ) -> NaiveCache:
        filtered_cache = NaiveCache(past_key_values.num_layers)
        for layer_idx in range(past_key_values.num_layers):
            key_states = past_key_values.key_cache[layer_idx]
            value_states = past_key_values.value_cache[layer_idx]
            if key_states is None or value_states is None:
                continue
            layer_keep = keep_indexes.to(device=key_states.device, dtype=torch.long)
            filtered_cache.key_cache[layer_idx] = key_states.index_select(0, layer_keep)
            filtered_cache.value_cache[layer_idx] = value_states.index_select(
                0, layer_keep
            )
        return filtered_cache

    def _select_context_for_vae_attention(
        self, gen_context: Dict[str, Any], allow_vae_condition: bool
    ) -> Dict[str, Any]:
        self._ensure_context_metadata(gen_context)
        if allow_vae_condition:
            return gen_context
        if len(gen_context["kv_lens"]) != 1:
            raise ValueError(
                "VAE-condition attention gating currently supports batch size = 1."
            )
        kv_len = int(gen_context["kv_lens"][0])
        vae_ranges = gen_context["vae_condition_ranges"][0]
        if kv_len <= 0 or len(vae_ranges) == 0:
            return gen_context
        device = self._model_device()
        keep_indexes = self._build_allowed_kv_indexes(
            kv_len=kv_len,
            vae_condition_ranges=vae_ranges,
            allow_vae_condition=False,
            device=device,
        )
        if keep_indexes.numel() == kv_len:
            return gen_context
        filtered_cache = self._filter_cache_with_indexes(
            gen_context["past_key_values"], keep_indexes
        )
        return {
            "past_key_values": filtered_cache,
            "kv_lens": [int(keep_indexes.numel())],
            "ropes": list(gen_context["ropes"]),
            "vae_condition_ranges": [[]],
        }

    def _forward_single_text_token(
        self,
        decode_state: Dict[str, Any],
        token_id: int,
        *,
        update_past_key_values: bool = True,
        allow_vae_condition: bool = True,
    ) -> torch.Tensor:
        if update_past_key_values and not allow_vae_condition:
            raise ValueError(
                "Updating decode_state with filtered VAE attention is not supported."
            )
        device = next(self.model.parameters()).device
        curr_tokens = torch.tensor([int(token_id)], dtype=torch.long, device=device)
        packed_text_embedding = self.model._embed_tokens(curr_tokens)
        query_lens = torch.ones_like(curr_tokens)

        kv_len = int(decode_state["kv_len"])
        vae_condition_ranges = decode_state.get("vae_condition_ranges", [])
        keep_indexes = self._build_allowed_kv_indexes(
            kv_len=kv_len,
            vae_condition_ranges=vae_condition_ranges,
            allow_vae_condition=allow_vae_condition,
            device=device,
        )
        effective_kv_len = int(keep_indexes.numel())
        key_values_lens = torch.tensor(
            [effective_kv_len], dtype=torch.int, device=device
        )
        packed_query_indexes = torch.tensor(
            [effective_kv_len], dtype=torch.long, device=device
        )
        packed_query_position_ids = torch.tensor(
            [int(decode_state["rope"])], dtype=torch.long, device=device
        )
        packed_key_value_indexes = torch.arange(
            effective_kv_len, dtype=torch.long, device=device
        )
        past_key_values = decode_state["past_key_values"]
        if not allow_vae_condition and len(vae_condition_ranges) > 0:
            past_key_values = self._filter_cache_with_indexes(
                past_key_values, keep_indexes
            )

        extra_inputs = {}
        if self.model.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.model.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=True,
            **extra_inputs,
        )
        if update_past_key_values:
            decode_state["past_key_values"] = output.past_key_values
            decode_state["kv_len"] = kv_len + 1
            decode_state["rope"] = int(decode_state["rope"]) + 1

        return output.packed_query_sequence

    def _forward_decode_text_token(
        self,
        decode_state: Dict[str, Any],
        token_id: int,
        text_can_attend_vae_condition: bool = True,
    ) -> torch.Tensor:
        """Forward a text token and update the KV cache.

        When ``text_can_attend_vae_condition=False``, the logits are computed
        from a forward pass where VAE-condition entries are filtered out of the
        KV cache (matching the training-time attention mask). However, the new
        KV entry stored back into the *original* (unfiltered) cache is the one
        produced by this filtered forward — so downstream VAE noise tokens that
        *do* attend to the full cache will see a text representation that was
        computed without peeking at VAE condition, exactly as in training.
        """
        vae_ranges = decode_state.get("vae_condition_ranges", [])
        if text_can_attend_vae_condition or len(vae_ranges) == 0:
            return self._forward_single_text_token(
                decode_state,
                token_id,
                update_past_key_values=True,
                allow_vae_condition=True,
            )

        # --- text should NOT attend to VAE condition blocks ---
        device = next(self.model.parameters()).device
        kv_len = int(decode_state["kv_len"])

        # 1. Build a filtered cache (deep copy, VAE-condition entries removed).
        keep_indexes = self._build_allowed_kv_indexes(
            kv_len=kv_len,
            vae_condition_ranges=vae_ranges,
            allow_vae_condition=False,
            device=device,
        )
        effective_kv_len = int(keep_indexes.numel())
        filtered_cache = self._filter_cache_with_indexes(
            decode_state["past_key_values"], keep_indexes
        )

        # 2. Forward through the model with the filtered cache, updating it.
        #    After this call, each layer of ``filtered_cache`` has
        #    ``effective_kv_len + 1`` entries (the +1 is the new text token).
        curr_tokens = torch.tensor([int(token_id)], dtype=torch.long, device=device)
        packed_text_embedding = self.model._embed_tokens(curr_tokens)
        query_lens = torch.ones_like(curr_tokens)
        key_values_lens = torch.tensor(
            [effective_kv_len], dtype=torch.int, device=device
        )
        packed_query_indexes = torch.tensor(
            [effective_kv_len], dtype=torch.long, device=device
        )
        packed_query_position_ids = torch.tensor(
            [int(decode_state["rope"])], dtype=torch.long, device=device
        )
        packed_key_value_indexes = torch.arange(
            effective_kv_len, dtype=torch.long, device=device
        )

        extra_inputs = {}
        if self.model.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.model.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=filtered_cache,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )

        # 3. Extract the new KV entry (last token) from the filtered cache and
        #    append it to the *original* unfiltered cache.
        original_cache = decode_state["past_key_values"]
        for layer_idx in range(original_cache.num_layers):
            filtered_key = filtered_cache.key_cache.get(layer_idx)
            filtered_value = filtered_cache.value_cache.get(layer_idx)
            if filtered_key is None:
                continue
            # The new token's KV is the last entry in the updated filtered cache.
            new_key = filtered_key[-1:]
            new_value = filtered_value[-1:]
            # Append to the original (unfiltered) cache.
            original_cache.key_cache[layer_idx] = torch.cat(
                [original_cache.key_cache[layer_idx], new_key], dim=0
            )
            original_cache.value_cache[layer_idx] = torch.cat(
                [original_cache.value_cache[layer_idx], new_value], dim=0
            )

        # 4. Update decode state.
        decode_state["past_key_values"] = original_cache
        decode_state["kv_len"] = kv_len + 1
        decode_state["rope"] = int(decode_state["rope"]) + 1

        return output.packed_query_sequence

    def _sample_next_token_id(
        self,
        packed_query_sequence: torch.Tensor,
        do_sample: bool,
        temperature: float,
        diagnostics: dict | None = None,
        action_token_id: int | None = None,
    ) -> int:
        pred_logits = self.model.language_model.lm_head(packed_query_sequence)
        if do_sample:
            safe_temp = max(float(temperature), 1e-6)
            probs = torch.softmax(pred_logits / safe_temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(pred_logits, dim=-1)
        next_id = int(next_token[0].item())

        # Log action token probability for diagnostics
        if diagnostics is not None and action_token_id is not None:
            with torch.no_grad():
                all_probs = torch.softmax(pred_logits.float(), dim=-1)
                action_prob = float(all_probs[0, action_token_id].item())
                # Track max action prob seen so far
                prev_max = diagnostics.get("action_token_max_prob", 0.0)
                diagnostics["action_token_max_prob"] = max(prev_max, action_prob)
                # Log details at the step where EOS is chosen
                eos_id = int(self.new_token_ids["eos_token_id"])
                if next_id == eos_id:
                    diagnostics["action_prob_at_eos"] = action_prob
                    # Rank of action token
                    action_logit = pred_logits[0, action_token_id]
                    rank = int((pred_logits[0] > action_logit).sum().item()) + 1
                    diagnostics["action_rank_at_eos"] = rank

        return next_id

    def _decode_text_until_stop_with_generate_text(
        self,
        decode_state: Dict[str, Any],
        max_length: int,
        do_sample: bool,
        text_temperature: float,
        stop_token_ids: List[int],
        start_token_id: int | None = None,
    ) -> Tuple[str, Optional[int], int]:
        generation_input = self.model.prepare_start_tokens(
            [int(decode_state["kv_len"])],
            [int(decode_state["rope"])],
            self.new_token_ids,
            start_token_id=start_token_id,
        )
        generation_input = self._to_model_device(generation_input)
        unpacked_latent, details = self.model.generate_text(
            past_key_values=decode_state["past_key_values"],
            max_length=max(int(max_length), 1),
            do_sample=do_sample,
            temperature=text_temperature,
            end_token_id=stop_token_ids,
            return_details=True,
            **generation_input,
        )
        token_ids = unpacked_latent[:, 0].tolist()
        bos_token_id = int(self.new_token_ids["bos_token_id"])
        # Strip BOS only when using the default start token (not custom inline)
        if start_token_id is None and token_ids and int(token_ids[0]) == bos_token_id:
            token_ids = token_ids[1:]
        text_segment = self.tokenizer.decode(token_ids).strip() if token_ids else ""

        num_steps = int(details.get("num_forward_steps", unpacked_latent.shape[0]))
        decode_state["past_key_values"] = details["past_key_values"]
        decode_state["kv_len"] = int(decode_state["kv_len"]) + num_steps
        decode_state["rope"] = int(decode_state["rope"]) + num_steps
        stop_token = details.get("stop_token_id")
        if stop_token is not None:
            stop_token = int(stop_token)
        return text_segment, stop_token, num_steps

    def _decode_interleaved_action_control(
        self,
        gen_context: Dict[str, Any],
        cfg_text_context: Dict[str, Any],
        cfg_img_context: Dict[str, Any],
        image_shapes,
        max_text_tokens: int,
        do_sample: bool,
        text_temperature: float,
        image_action_token_id: int,
        max_auto_images: Optional[int],
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval,
        timestep_shift: float,
        num_timesteps: int,
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        enable_taylorseer: bool,
        feedback_generated_image_to_context: bool,
        feedback_generated_image_as_vae_condition: bool,
        feedback_generated_image_as_vit_condition: bool,
        diagnostics: Optional[Dict[str, Any]] = None,
        enable_size_tokens: bool = False,
        enable_ref_image: bool = False,
        text_can_attend_vae_condition: bool = True,
        ref_target_can_attend_vae_condition: bool = False,
        no_ref_target_can_attend_vae_condition: bool = True,
        all_images: Optional[List[Union[Image.Image, torch.Tensor]]] = None,
        prefill_assistant_text: Optional[str] = None,
        inline_image_latents: bool = True,
    ) -> List[Union[str, Image.Image, Dict[str, Any]]]:

        output_list = []
        if all_images is None:
            all_images = []
        eos_token_id = int(self.new_token_ids["eos_token_id"])
        bos_token_id = int(self.new_token_ids["bos_token_id"])
        decode_state = self._init_single_decode_state(gen_context)
        curr_token_id = bos_token_id
        segment_token_ids: List[int] = []
        decoded_steps = 0
        image_count = 0
        image_limit: Optional[int] = None
        if max_auto_images is not None:
            parsed_limit = int(max_auto_images)
            if parsed_limit > 0:
                image_limit = parsed_limit
        if diagnostics is not None:
            text_segments_so_far = int(diagnostics.get("text_segments", 0))
            image_outputs_so_far = int(diagnostics.get("image_outputs", 0))
            image_attempts_so_far = int(diagnostics.get("image_generation_attempts", 0))
            image_success_so_far = int(diagnostics.get("image_generation_success", 0))
            feedback_so_far = int(diagnostics.get("image_feedback_applied", 0))
            diagnostics.update(
                {
                    "mode": "action_token",
                    "max_text_tokens": int(max(int(max_text_tokens), 1)),
                    "image_action_token_id": int(image_action_token_id),
                    "max_auto_images": max_auto_images,
                    "image_limit_active": image_limit is not None,
                    "decode_steps": 0,
                    "action_token_hits": 0,
                    "text_segments": text_segments_so_far,
                    "image_outputs": image_outputs_so_far,
                    "image_generation_attempts": image_attempts_so_far,
                    "image_generation_success": image_success_so_far,
                    "image_feedback_applied": feedback_so_far,
                    "predicted_image_sizes": [],
                    "stop_reason": "max_text_tokens",
                }
            )

        # ---- Prefill: inject custom text and generate image directly ----
        if prefill_assistant_text is not None:
            # 1. Output the prefill text as a text segment
            output_list.append(prefill_assistant_text)
            if diagnostics is not None:
                diagnostics["text_segments"] += 1
                diagnostics["action_token_hits"] += 1

            # 2. Build full assistant turn: reasoning + action_token + size_tokens
            #    update_context_text wraps as [BOS ... EOS], matching training format
            if isinstance(image_shapes, (list, tuple)):
                h, w = int(image_shapes[0]), int(image_shapes[1])
            else:
                h = w = int(image_shapes)
            action_str = self.tokenizer.decode([image_action_token_id])
            full_turn_text = f"{prefill_assistant_text}{action_str}({h}, {w})"
            gen_context = self.update_context_text(full_turn_text, gen_context)
            cfg_img_context = self.update_context_text(
                prefill_assistant_text, cfg_img_context
            )
            if diagnostics is not None:
                diagnostics["predicted_image_sizes"].append(
                    {"size": (h, w), "source": "prefill"}
                )

            # 3. Generate image
            if diagnostics is not None:
                diagnostics["image_generation_attempts"] += 1
            image_can_attend_vae_condition = no_ref_target_can_attend_vae_condition
            if diagnostics is not None:
                diagnostics["image_can_attend_vae_condition"] = bool(
                    image_can_attend_vae_condition
                )
            img = self.gen_image(
                (h, w),
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                enable_taylorseer=enable_taylorseer,
                can_attend_vae_condition=image_can_attend_vae_condition,
            )
            output_list.append(img)
            image_count += 1
            if diagnostics is not None:
                diagnostics["image_outputs"] += 1
                generated_image = img.get("image") if isinstance(img, dict) else img
                diagnostics["image_generation_success"] += int(
                    generated_image is not None
                )

            generated_image = img.get("image") if isinstance(img, dict) else img
            if generated_image is not None:
                all_images.append(self._to_pil_image(generated_image))

            # 4. Feed generated image back to context
            feedback_use_vae = feedback_generated_image_as_vae_condition
            feedback_use_vit = feedback_generated_image_as_vit_condition
            if feedback_generated_image_to_context and (
                feedback_use_vae or feedback_use_vit
            ):
                if generated_image is not None:
                    generated_image = self._to_pil_image(generated_image)
                    generated_image = self.vae_transform.resize_transform(
                        generated_image
                    )
                    gen_context = self.update_context_image(
                        generated_image,
                        gen_context,
                        vae=feedback_use_vae,
                        vit=feedback_use_vit,
                    )
                    if diagnostics is not None:
                        diagnostics["image_feedback_applied"] += 1
                    image_shapes = generated_image.size[::-1]

            # 5. Reset contexts and decode state for post-image text generation
            cfg_text_context = deepcopy(gen_context)
            cfg_img_context = deepcopy(gen_context)
            decode_state = self._init_single_decode_state(gen_context)
            curr_token_id = bos_token_id
        # ---- End prefill handling ----

        inline_start_token_id = None  # for Path A inline continuation
        max_decode_steps = max(int(max_text_tokens), 1)
        while decoded_steps < max_decode_steps:
            if text_can_attend_vae_condition:
                remaining_steps = max_decode_steps - decoded_steps
                text_segment, stop_token_id, used_steps = (
                    self._decode_text_until_stop_with_generate_text(
                        decode_state=decode_state,
                        max_length=remaining_steps,
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        stop_token_ids=[eos_token_id, image_action_token_id],
                        start_token_id=inline_start_token_id,
                    )
                )
                inline_start_token_id = None  # consumed
                decoded_steps += used_steps
                if diagnostics is not None:
                    diagnostics["decode_steps"] += used_steps

                if text_segment:
                    output_list.append(text_segment)
                    if diagnostics is not None:
                        diagnostics["text_segments"] += 1
                    cfg_img_context = self.update_context_text(
                        text_segment, cfg_img_context
                    )

                if stop_token_id is None:
                    if diagnostics is not None:
                        diagnostics["stop_reason"] = "max_text_tokens"
                    break

                if stop_token_id == eos_token_id:
                    self._forward_single_text_token(decode_state, eos_token_id)
                    self._sync_decode_state_to_context(decode_state, gen_context)
                    if diagnostics is not None:
                        diagnostics["stop_reason"] = "eos_token"
                    break

                next_token_id = int(stop_token_id)
            else:
                if diagnostics is not None:
                    diagnostics["decode_steps"] += 1
                packed_query_sequence = self._forward_decode_text_token(
                    decode_state,
                    curr_token_id,
                    text_can_attend_vae_condition=text_can_attend_vae_condition,
                )
                next_token_id = self._sample_next_token_id(
                    packed_query_sequence,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    diagnostics=diagnostics,
                    action_token_id=image_action_token_id,
                )
                decoded_steps += 1

            if next_token_id == eos_token_id:
                # With inline action tokens, the training structure is:
                #   [BOS text... action_token EOS] [image] [BOS text... EOS]
                # If the model wanted an image, it would have emitted
                # action_token before EOS (caught above).  EOS without a
                # preceding action token means the response is complete.
                text_segment = self.tokenizer.decode(segment_token_ids).strip()
                if text_segment:
                    output_list.append(text_segment)
                    if diagnostics is not None:
                        diagnostics["text_segments"] += 1
                    cfg_img_context = self.update_context_text(
                        text_segment, cfg_img_context
                    )
                segment_token_ids = []
                self._forward_single_text_token(decode_state, eos_token_id)
                self._sync_decode_state_to_context(decode_state, gen_context)
                if diagnostics is not None:
                    diagnostics["stop_reason"] = "eos_token"
                break

            if next_token_id == image_action_token_id:
                if diagnostics is not None:
                    diagnostics["action_token_hits"] += 1
                if not text_can_attend_vae_condition:
                    text_segment = self.tokenizer.decode(segment_token_ids).strip()
                    if text_segment:
                        output_list.append(text_segment)
                        if diagnostics is not None:
                            diagnostics["text_segments"] += 1
                        cfg_img_context = self.update_context_text(
                            text_segment, cfg_img_context
                        )
                segment_token_ids = []

                # Training structure (inline action token):
                #   [BOS text... action_token (H, W) EOS]   (with size tokens)
                #   [BOS text... action_token EOS]           (without)
                # The action token was just predicted; feed it, optionally
                # decode size tokens, then close with EOS.
                packed_query = self._forward_decode_text_token(
                    decode_state,
                    image_action_token_id,
                    text_can_attend_vae_condition=text_can_attend_vae_condition,
                )
                ref_image_idx = None
                if enable_size_tokens:
                    parsed, packed_query = self._decode_size_tokens(
                        decode_state,
                        packed_query,
                        text_can_attend_vae_condition=text_can_attend_vae_condition,
                    )
                    if parsed is not None:
                        image_shapes = self._clamp_image_size(parsed)
                        if diagnostics is not None:
                            diagnostics["predicted_image_size"] = parsed
                            diagnostics["predicted_image_sizes"].append(
                                {"size": parsed, "source": "predicted"}
                            )
                    else:
                        if diagnostics is not None:
                            diagnostics["predicted_image_sizes"].append(
                                {
                                    "size": tuple(image_shapes)
                                    if isinstance(image_shapes, list)
                                    else image_shapes,
                                    "source": "default",
                                }
                            )
                if enable_ref_image:
                    ref_image_idx, packed_query = self._decode_ref_tokens(
                        decode_state,
                        packed_query,
                        text_can_attend_vae_condition=text_can_attend_vae_condition,
                    )
                    if diagnostics is not None and ref_image_idx is not None:
                        diagnostics["predicted_ref_index"] = int(ref_image_idx)
                if not inline_image_latents:
                    self._forward_single_text_token(decode_state, eos_token_id)
                self._sync_decode_state_to_context(decode_state, gen_context)

                if image_limit is not None and image_count >= image_limit:
                    if diagnostics is not None:
                        diagnostics["stop_reason"] = "image_limit"
                    break

                if diagnostics is not None:
                    diagnostics["image_generation_attempts"] += 1
                ref_latent = None
                if (
                    enable_ref_image
                    and ref_image_idx is not None
                    and 0 <= ref_image_idx < len(all_images)
                ):
                    ref_latent = self._encode_ref_latent(
                        all_images[ref_image_idx], image_shapes
                    )
                    if diagnostics is not None:
                        diagnostics["resolved_ref_index"] = int(ref_image_idx)
                image_can_attend_vae_condition = (
                    ref_target_can_attend_vae_condition
                    if ref_latent is not None
                    else no_ref_target_can_attend_vae_condition
                )
                if diagnostics is not None:
                    diagnostics["image_can_attend_vae_condition"] = bool(
                        image_can_attend_vae_condition
                    )
                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                    ref_latent=ref_latent,
                    can_attend_vae_condition=image_can_attend_vae_condition,
                )
                output_list.append(img)
                image_count += 1
                if diagnostics is not None:
                    diagnostics["image_outputs"] += 1
                    generated_image = img.get("image") if isinstance(img, dict) else img
                    diagnostics["image_generation_success"] += int(
                        generated_image is not None
                    )

                generated_image = img.get("image") if isinstance(img, dict) else img
                if generated_image is not None:
                    all_images.append(self._to_pil_image(generated_image))

                feedback_use_vae = feedback_generated_image_as_vae_condition
                feedback_use_vit = feedback_generated_image_as_vit_condition
                inline_last_hidden = None
                if feedback_generated_image_to_context and (
                    feedback_use_vae or feedback_use_vit
                ):
                    if generated_image is not None:
                        generated_image = self._to_pil_image(generated_image)
                        generated_image = self.vae_transform.resize_transform(
                            generated_image
                        )
                        if inline_image_latents and feedback_use_vit:
                            gen_context, inline_last_hidden = (
                                self.update_context_image(
                                    generated_image,
                                    gen_context,
                                    vae=feedback_use_vae,
                                    vit=feedback_use_vit,
                                    return_last_hidden_state=True,
                                )
                            )
                        else:
                            gen_context = self.update_context_image(
                                generated_image,
                                gen_context,
                                vae=feedback_use_vae,
                                vit=feedback_use_vit,
                            )
                        if diagnostics is not None:
                            diagnostics["image_feedback_applied"] += 1
                        image_shapes = generated_image.size[::-1]

                cfg_text_context = deepcopy(gen_context)
                cfg_img_context = deepcopy(gen_context)
                decode_state = self._init_single_decode_state(gen_context)

                if inline_image_latents and inline_last_hidden is not None:
                    # Sample first continuation token from ViT last hidden state
                    sampled_id = self._sample_next_token_id(
                        inline_last_hidden,
                        do_sample=do_sample,
                        temperature=text_temperature,
                    )
                    if text_can_attend_vae_condition:
                        # Path A: pass as start_token_id to generate_text
                        inline_start_token_id = sampled_id
                    else:
                        # Path B: use as curr_token_id for token-by-token decode
                        curr_token_id = sampled_id
                        segment_token_ids.append(sampled_id)
                else:
                    curr_token_id = bos_token_id
                continue

            if text_can_attend_vae_condition:
                if diagnostics is not None:
                    diagnostics["stop_reason"] = "max_text_tokens"
                break

            segment_token_ids.append(next_token_id)
            curr_token_id = next_token_id

        if segment_token_ids:
            text_segment = self.tokenizer.decode(segment_token_ids).strip()
            if text_segment:
                output_list.append(text_segment)
                if diagnostics is not None:
                    diagnostics["text_segments"] += 1
        self._sync_decode_state_to_context(decode_state, gen_context)
        if diagnostics is not None:
            diagnostics["output_items"] = len(output_list)
            decode_steps = diagnostics.get("decode_steps", 0)
            if decode_steps > 0:
                diagnostics["action_token_hit_rate"] = (
                    diagnostics.get("action_token_hits", 0) / decode_steps
                )
            else:
                diagnostics["action_token_hit_rate"] = 0.0

        return output_list

    def _decode_interleaved_action_control_parse_mode(
        self,
        gen_context: Dict[str, Any],
        cfg_text_context: Dict[str, Any],
        cfg_img_context: Dict[str, Any],
        image_shapes,
        max_text_tokens: int,
        do_sample: bool,
        text_temperature: float,
        image_action_token: str,
        max_auto_images: Optional[int],
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval,
        timestep_shift: float,
        num_timesteps: int,
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        enable_taylorseer: bool,
        feedback_generated_image_to_context: bool,
        feedback_generated_image_as_vae_condition: bool,
        feedback_generated_image_as_vit_condition: bool,
        diagnostics: Optional[Dict[str, Any]] = None,
        enable_size_tokens: bool = False,
        enable_ref_image: bool = False,
        text_can_attend_vae_condition: bool = True,
        ref_target_can_attend_vae_condition: bool = False,
        no_ref_target_can_attend_vae_condition: bool = True,
        all_images: Optional[List[Union[Image.Image, torch.Tensor]]] = None,
        prefill_assistant_text: Optional[str] = None,
    ) -> List[Union[str, Image.Image, Dict[str, Any]]]:
        """Stepwise plain-text action decoding.

        Unlike the original COOPER-style "generate then parse" approach, this
        decoder stops as soon as the plain-text action pattern is fully formed,
        generates the image, feeds it back into context, and only then resumes
        decoding later text. This matches the training-time interleaving more
        closely for sequences like ``thought -> image -> thought -> answer``.
        """
        output_list: List[Union[str, Image.Image, Dict[str, Any]]] = []
        if all_images is None:
            all_images = []
        eos_token_id = int(self.new_token_ids["eos_token_id"])
        bos_token_id = int(self.new_token_ids["bos_token_id"])
        decode_state = self._init_single_decode_state(gen_context)
        curr_token_id = bos_token_id
        segment_token_ids: List[int] = []
        decoded_steps = 0
        image_count = 0
        image_limit: Optional[int] = None
        if max_auto_images is not None:
            parsed_limit = int(max_auto_images)
            if parsed_limit > 0:
                image_limit = parsed_limit

        # Regex to find action token, with optional or required size spec
        if enable_size_tokens:
            action_pattern = re.compile(
                re.escape(image_action_token) + r"\s*\((\d+),\s*(\d+)\)"
            )
        else:
            action_pattern = re.compile(re.escape(image_action_token))

        if enable_ref_image:
            raise NotImplementedError(
                "enable_ref_image is not yet supported with plain-text action token "
                "parse mode (register_action_token=false). Use register_action_token=true "
                "for ref-image support."
            )

        if diagnostics is not None:
            diagnostics.update(
                {
                    "mode": "action_token_parse",
                    "max_text_tokens": int(max(int(max_text_tokens), 1)),
                    "max_auto_images": max_auto_images,
                    "image_limit_active": image_limit is not None,
                    "decode_steps": 0,
                    "action_token_hits": 0,
                    "predicted_image_sizes": [],
                    "stop_reason": "eos_token",
                }
            )

        # ---- Prefill handling ----
        if prefill_assistant_text is not None:
            output_list.append(prefill_assistant_text)
            if diagnostics is not None:
                diagnostics["text_segments"] = diagnostics.get("text_segments", 0) + 1
                diagnostics["action_token_hits"] += 1
            if isinstance(image_shapes, (list, tuple)):
                h, w = int(image_shapes[0]), int(image_shapes[1])
            else:
                h = w = int(image_shapes)
            full_turn_text = f"{prefill_assistant_text}{image_action_token}({h}, {w})"
            gen_context = self.update_context_text(full_turn_text, gen_context)
            cfg_img_context = self.update_context_text(
                prefill_assistant_text, cfg_img_context
            )
            if diagnostics is not None:
                diagnostics["predicted_image_sizes"].append(
                    {"size": (h, w), "source": "prefill"}
                )
                diagnostics["image_generation_attempts"] = (
                    diagnostics.get("image_generation_attempts", 0) + 1
                )
            img = self.gen_image(
                (h, w),
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                enable_taylorseer=enable_taylorseer,
                can_attend_vae_condition=no_ref_target_can_attend_vae_condition,
            )
            output_list.append(img)
            image_count += 1
            if diagnostics is not None:
                diagnostics["image_outputs"] = diagnostics.get("image_outputs", 0) + 1
                generated_image = img.get("image") if isinstance(img, dict) else img
                diagnostics["image_generation_success"] = diagnostics.get(
                    "image_generation_success", 0
                ) + int(generated_image is not None)
            generated_image = img.get("image") if isinstance(img, dict) else img
            if generated_image is not None:
                all_images.append(self._to_pil_image(generated_image))
            feedback_use_vae = feedback_generated_image_as_vae_condition
            feedback_use_vit = feedback_generated_image_as_vit_condition
            if feedback_generated_image_to_context and (
                feedback_use_vae or feedback_use_vit
            ):
                if generated_image is not None:
                    generated_image = self._to_pil_image(generated_image)
                    generated_image = self.vae_transform.resize_transform(
                        generated_image
                    )
                    gen_context = self.update_context_image(
                        generated_image,
                        gen_context,
                        vae=feedback_use_vae,
                        vit=feedback_use_vit,
                    )
                    if diagnostics is not None:
                        diagnostics["image_feedback_applied"] = (
                            diagnostics.get("image_feedback_applied", 0) + 1
                        )
                    image_shapes = generated_image.size[::-1]
            cfg_text_context = deepcopy(gen_context)
            cfg_img_context = deepcopy(gen_context)
            decode_state = self._init_single_decode_state(gen_context)
            curr_token_id = bos_token_id
        # ---- End prefill ----

        max_decode_steps = max(int(max_text_tokens), 1)
        while decoded_steps < max_decode_steps:
            if diagnostics is not None:
                diagnostics["decode_steps"] += 1
            packed_query_sequence = self._forward_decode_text_token(
                decode_state,
                curr_token_id,
                text_can_attend_vae_condition=text_can_attend_vae_condition,
            )
            next_token_id = self._sample_next_token_id(
                packed_query_sequence,
                do_sample=do_sample,
                temperature=text_temperature,
            )
            decoded_steps += 1

            if next_token_id == eos_token_id:
                text_segment = self.tokenizer.decode(segment_token_ids).strip()
                if text_segment:
                    output_list.append(text_segment)
                    if diagnostics is not None:
                        diagnostics["text_segments"] = (
                            diagnostics.get("text_segments", 0) + 1
                        )
                self._forward_single_text_token(decode_state, eos_token_id)
                self._sync_decode_state_to_context(decode_state, gen_context)
                if diagnostics is not None:
                    diagnostics["stop_reason"] = "eos_token"
                break

            segment_token_ids.append(next_token_id)
            curr_token_id = next_token_id
            decoded_text = self.tokenizer.decode(segment_token_ids)
            match = action_pattern.search(decoded_text)
            if match is None:
                continue

            pre_text = decoded_text[: match.start()].strip()
            if pre_text:
                output_list.append(pre_text)
                if diagnostics is not None:
                    diagnostics["text_segments"] = (
                        diagnostics.get("text_segments", 0) + 1
                    )

            if diagnostics is not None:
                diagnostics["action_token_hits"] += 1

            if image_limit is not None and image_count >= image_limit:
                if diagnostics is not None:
                    diagnostics["stop_reason"] = "image_limit"
                post_text = decoded_text[match.end() :].strip()
                if post_text:
                    output_list.append(post_text)
                    if diagnostics is not None:
                        diagnostics["text_segments"] = (
                            diagnostics.get("text_segments", 0) + 1
                        )
                break

            if enable_size_tokens and match.lastindex and match.lastindex >= 2:
                h, w = int(match.group(1)), int(match.group(2))
            else:
                if isinstance(image_shapes, (list, tuple)):
                    h, w = int(image_shapes[0]), int(image_shapes[1])
                else:
                    h = w = int(image_shapes)
            if diagnostics is not None:
                diagnostics["predicted_image_sizes"].append(
                    {
                        "size": (h, w),
                        "source": "parsed" if enable_size_tokens else "default",
                    }
                )

            context_text = decoded_text[: match.end()]
            gen_context = self.update_context_text(context_text, gen_context)
            cfg_img_context = self.update_context_text(
                decoded_text[: match.start()], cfg_img_context
            )

            if diagnostics is not None:
                diagnostics["image_generation_attempts"] = (
                    diagnostics.get("image_generation_attempts", 0) + 1
                )
            img = self.gen_image(
                (h, w),
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                enable_taylorseer=enable_taylorseer,
                can_attend_vae_condition=no_ref_target_can_attend_vae_condition,
            )
            output_list.append(img)
            image_count += 1
            if diagnostics is not None:
                diagnostics["image_outputs"] = diagnostics.get("image_outputs", 0) + 1
                generated_image = img.get("image") if isinstance(img, dict) else img
                diagnostics["image_generation_success"] = diagnostics.get(
                    "image_generation_success", 0
                ) + int(generated_image is not None)

            generated_image = img.get("image") if isinstance(img, dict) else img
            if generated_image is not None:
                all_images.append(self._to_pil_image(generated_image))

            feedback_use_vae = feedback_generated_image_as_vae_condition
            feedback_use_vit = feedback_generated_image_as_vit_condition
            if feedback_generated_image_to_context and (
                feedback_use_vae or feedback_use_vit
            ):
                if generated_image is not None:
                    generated_image = self._to_pil_image(generated_image)
                    generated_image = self.vae_transform.resize_transform(
                        generated_image
                    )
                    gen_context = self.update_context_image(
                        generated_image,
                        gen_context,
                        vae=feedback_use_vae,
                        vit=feedback_use_vit,
                    )
                    if diagnostics is not None:
                        diagnostics["image_feedback_applied"] = (
                            diagnostics.get("image_feedback_applied", 0) + 1
                        )
                    image_shapes = generated_image.size[::-1]

            cfg_text_context = deepcopy(gen_context)
            cfg_img_context = deepcopy(gen_context)
            decode_state = self._init_single_decode_state(gen_context)
            curr_token_id = bos_token_id
            segment_token_ids = []
        else:
            if diagnostics is not None:
                diagnostics["stop_reason"] = "max_text_tokens"

        if diagnostics is not None:
            diagnostics["output_items"] = len(output_list)

        return output_list

    def init_gen_context(self):
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(
                self.model.config.llm_config.num_hidden_layers
            ),
            "vae_condition_ranges": [[]],
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference,
        self._ensure_context_metadata(gen_context)
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._to_model_device(generation_input)

        past_key_values = self.model.forward_cache_update_text(
            past_key_values, **generation_input
        )
        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values

        return gen_context

    @torch.no_grad()
    def update_context_image(
        self, image, gen_context, vae=True, vit=True,
        return_last_hidden_state=False,
    ):
        # used for interleave data, currently only support 1 data inference,
        self._ensure_context_metadata(gen_context)
        assert vae or vit
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        last_hidden_state = None

        if vae:
            ## update vae
            prev_kv_lens = [int(x) for x in kv_lens]
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = self._to_model_device(generation_input)
            past_key_values = self.model.forward_cache_update_vae(
                self.vae_model, past_key_values, **generation_input
            )
            for idx, (start, end) in enumerate(zip(prev_kv_lens, kv_lens)):
                start = int(start)
                end = int(end)
                if end > start:
                    gen_context["vae_condition_ranges"][idx].append((start, end))

        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = self._to_model_device(generation_input)
            if return_last_hidden_state:
                past_key_values, last_hidden_state = (
                    self.model.forward_cache_update_vit(
                        past_key_values,
                        return_last_hidden_state=True,
                        **generation_input,
                    )
                )
            else:
                past_key_values = self.model.forward_cache_update_vit(
                    past_key_values, **generation_input
                )

        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values

        if return_last_hidden_state:
            return gen_context, last_hidden_state
        return gen_context

    @torch.no_grad()
    def _encode_ref_latent(
        self, image: Union[Image.Image, torch.Tensor], image_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Encode a reference image into VAE latent aligned to target ``image_shape``."""
        ref_pil = self._to_pil_image(image).resize(
            (image_shape[1], image_shape[0]), Image.LANCZOS
        )
        ref_tensor = self.vae_transform.to_tensor_transform(ref_pil)
        ref_tensor = self.vae_transform.normalize_transform(ref_tensor).unsqueeze(0)
        device = self._model_device()
        vae_dtype = next(self.vae_model.parameters()).dtype
        ref_tensor = ref_tensor.to(device=device, dtype=vae_dtype)
        return self.vae_model.encode(ref_tensor)

    # ------------------------------------------------------------------ #
    #  Multi-GPU CFG parallel helpers
    # ------------------------------------------------------------------ #

    def _copy_naive_cache_to_device(self, past_key_values, device):
        copied = NaiveCache(past_key_values.num_layers)
        for layer_idx in range(past_key_values.num_layers):
            key_cache = past_key_values.key_cache[layer_idx]
            if key_cache is None:
                continue
            copied.key_cache[layer_idx] = key_cache.to(device, non_blocking=True)
            copied.value_cache[layer_idx] = past_key_values.value_cache[
                layer_idx
            ].to(device, non_blocking=True)
        return copied

    def _move_generation_input_to_device(self, generation_input, device, skip_keys=None):
        moved = {}
        skip_keys = skip_keys or set()
        for key, value in generation_input.items():
            if key in skip_keys or not torch.is_tensor(value):
                moved[key] = value
            else:
                moved[key] = value.to(device, non_blocking=True)
        return moved

    def _can_use_cfg_parallel(self, cfg_text_scale, cfg_img_scale):
        if not self.cfg_parallel_workers:
            return False
        required_workers = ["base"]
        if cfg_text_scale > 1.0:
            required_workers.append("cfg_text")
        if cfg_img_scale > 1.0:
            required_workers.append("cfg_img")
        return all(w in self.cfg_parallel_workers for w in required_workers)

    def _build_cfg_branch_state(
        self,
        worker_name,
        image_generation_input,
        branch_context,
        packed_query_position_ids,
        packed_query_indexes,
        key_values_lens,
        packed_key_value_indexes,
    ):
        worker = self.cfg_parallel_workers[worker_name]
        device = worker["device"]
        branch_state = self._move_generation_input_to_device(
            image_generation_input, device, skip_keys={"packed_init_noises"},
        )
        branch_state["device"] = device
        branch_state["model"] = worker["model"]
        branch_state["past_key_values"] = self._copy_naive_cache_to_device(
            branch_context["past_key_values"], device
        )
        branch_state["packed_position_ids"] = packed_query_position_ids.to(
            device, non_blocking=True
        )
        branch_state["packed_indexes"] = packed_query_indexes.to(
            device, non_blocking=True
        )
        branch_state["key_values_lens"] = key_values_lens.to(device, non_blocking=True)
        branch_state["packed_key_value_indexes"] = packed_key_value_indexes.to(
            device, non_blocking=True
        )
        return branch_state

    def _run_cfg_branch_step(self, branch_state, x_t, timestep):
        branch_state["model"].language_model.model.enable_taylorseer = False
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            return branch_state["model"]._forward_flow(
                x_t=x_t.to(branch_state["device"], non_blocking=True),
                timestep=timestep.to(branch_state["device"], non_blocking=True),
                packed_vae_token_indexes=branch_state["packed_vae_token_indexes"],
                packed_vae_position_ids=branch_state["packed_vae_position_ids"],
                packed_text_ids=branch_state["packed_text_ids"],
                packed_text_indexes=branch_state["packed_text_indexes"],
                packed_indexes=branch_state["packed_indexes"],
                packed_position_ids=branch_state["packed_position_ids"],
                packed_seqlens=branch_state["packed_seqlens"],
                key_values_lens=branch_state["key_values_lens"],
                past_key_values=branch_state["past_key_values"],
                packed_key_value_indexes=branch_state["packed_key_value_indexes"],
                cfg_text_scale=1.0,
                cfg_img_scale=1.0,
                cfg_type="sequential",
            )

    def _apply_cfg_guidance(
        self, v_t, cfg_text_v_t, cfg_img_v_t,
        cfg_text_scale, cfg_img_scale, cfg_renorm_min, cfg_renorm_type,
    ):
        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(
                    min=cfg_renorm_min, max=1.0
                )
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    return cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                return v_t_text

            v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
            if cfg_img_scale > 1.0:
                v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
            else:
                v_t_ = v_t_text_

            if cfg_renorm_type == "global":
                norm_v_t = torch.norm(v_t)
                norm_v_t_ = torch.norm(v_t_)
            elif cfg_renorm_type == "channel":
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
            else:
                raise NotImplementedError(f"{cfg_renorm_type} is not supported")

            scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(
                min=cfg_renorm_min, max=1.0
            )
            return v_t_ * scale

        return v_t

    # ------------------------------------------------------------------ #
    #  Multi-GPU parallel image generation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def gen_image_parallel(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=None,
        enable_taylorseer=False,
        ref_latent=None,
        can_attend_vae_condition=True,
    ):
        """Image generation using multi-GPU CFG parallel denoising."""
        if enable_taylorseer:
            raise NotImplementedError(
                "TaylorSeer is not supported in multi-GPU CFG parallel mode."
            )
        if timestep_shift is None:
            timestep_shift = float(getattr(self.model.config, "timestep_shift", 1.0))

        base_worker = self.cfg_parallel_workers["base"]
        base_model = base_worker["model"]
        base_device = base_worker["device"]

        # Prepare generation inputs (same as gen_image sequential path)
        selected_context = self._select_context_for_vae_attention(
            gen_context, allow_vae_condition=can_attend_vae_condition
        )
        image_generation_input = base_model.prepare_vae_latent(
            curr_kvlens=selected_context["kv_lens"],
            curr_rope=selected_context["ropes"],
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
            ref_latents=[ref_latent],
        )
        image_generation_input = self._move_generation_input_to_device(
            image_generation_input, base_device
        )

        cfg_text_context = self._select_context_for_vae_attention(
            cfg_text_precontext, allow_vae_condition=can_attend_vae_condition
        )
        generation_input_cfg_text = base_model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=cfg_text_context["ropes"],
            image_sizes=[image_shape],
        )
        generation_input_cfg_text = self._move_generation_input_to_device(
            generation_input_cfg_text, base_device
        )

        cfg_img_context = self._select_context_for_vae_attention(
            cfg_img_precontext, allow_vae_condition=can_attend_vae_condition
        )
        generation_input_cfg_img = base_model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_context["kv_lens"],
            curr_rope=cfg_img_context["ropes"],
            image_sizes=[image_shape],
        )
        generation_input_cfg_img = self._move_generation_input_to_device(
            generation_input_cfg_img, base_device
        )

        # Build branch states
        branch_states = {
            "base": self._build_cfg_branch_state(
                "base",
                image_generation_input,
                selected_context,
                image_generation_input["packed_position_ids"],
                image_generation_input["packed_indexes"],
                image_generation_input["key_values_lens"],
                image_generation_input["packed_key_value_indexes"],
            )
        }
        if cfg_text_scale > 1.0:
            branch_states["cfg_text"] = self._build_cfg_branch_state(
                "cfg_text",
                image_generation_input,
                cfg_text_context,
                generation_input_cfg_text["cfg_packed_position_ids"],
                generation_input_cfg_text["cfg_packed_query_indexes"],
                generation_input_cfg_text["cfg_key_values_lens"],
                generation_input_cfg_text["cfg_packed_key_value_indexes"],
            )
        if cfg_img_scale > 1.0:
            branch_states["cfg_img"] = self._build_cfg_branch_state(
                "cfg_img",
                image_generation_input,
                cfg_img_context,
                generation_input_cfg_img["cfg_packed_position_ids"],
                generation_input_cfg_img["cfg_packed_query_indexes"],
                generation_input_cfg_img["cfg_key_values_lens"],
                generation_input_cfg_img["cfg_packed_key_value_indexes"],
            )

        # Denoising loop
        from tqdm import tqdm
        x_t = image_generation_input["packed_init_noises"].to(
            base_device, non_blocking=True
        )
        timesteps = torch.linspace(1, 0, num_timesteps, device=base_device)
        timesteps = timestep_shift * timesteps / (
            1 + (timestep_shift - 1) * timesteps
        )
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(
            enumerate(timesteps), total=len(timesteps), desc="Generating Images",
        ):
            timestep = torch.full((x_t.shape[0],), t, device=base_device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0

            active_branch_names = ["base"]
            if cfg_text_scale_ > 1.0:
                active_branch_names.append("cfg_text")
            if cfg_img_scale_ > 1.0:
                active_branch_names.append("cfg_img")

            if self.cfg_parallel_executor is None or len(active_branch_names) == 1:
                branch_outputs = {
                    name: self._run_cfg_branch_step(
                        branch_states[name], x_t, timestep
                    )
                    for name in active_branch_names
                }
            else:
                futures = {
                    name: self.cfg_parallel_executor.submit(
                        self._run_cfg_branch_step,
                        branch_states[name],
                        x_t,
                        timestep,
                    )
                    for name in active_branch_names
                }
                branch_outputs = {
                    name: future.result() for name, future in futures.items()
                }

            v_t = branch_outputs["base"].to(base_device, non_blocking=True)
            cfg_text_v_t = None
            cfg_img_v_t = None
            if cfg_text_scale_ > 1.0:
                cfg_text_v_t = branch_outputs["cfg_text"].to(
                    base_device, non_blocking=True
                )
            if cfg_img_scale_ > 1.0:
                cfg_img_v_t = branch_outputs["cfg_img"].to(
                    base_device, non_blocking=True
                )

            v_t = self._apply_cfg_guidance(
                v_t, cfg_text_v_t, cfg_img_v_t,
                cfg_text_scale_, cfg_img_scale_,
                cfg_renorm_min, cfg_renorm_type,
            )
            x_t = x_t - v_t.to(x_t.device) * dts[i]

        # Unpack and decode
        sample_token_lens = (branch_states["base"]["packed_seqlens"] - 2).tolist()
        unpacked_latent = x_t.split(sample_token_lens)
        image = self.decode_image(unpacked_latent[0], image_shape, return_tensor=False)
        return image

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift: Optional[float] = None,
        enable_taylorseer=False,
        ref_latent: Optional[torch.Tensor] = None,
        can_attend_vae_condition: bool = True,
    ):
        # Dispatch to multi-GPU parallel path if workers are available
        if self._can_use_cfg_parallel(cfg_text_scale, cfg_img_scale):
            return self.gen_image_parallel(
                image_shape=image_shape, gen_context=gen_context,
                cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img_scale,
                cfg_text_precontext=cfg_text_precontext,
                cfg_img_precontext=cfg_img_precontext,
                cfg_interval=cfg_interval, cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type, num_timesteps=num_timesteps,
                timestep_shift=timestep_shift, enable_taylorseer=enable_taylorseer,
                ref_latent=ref_latent, can_attend_vae_condition=can_attend_vae_condition,
            )

        if timestep_shift is None:
            timestep_shift = float(getattr(self.model.config, "timestep_shift", 1.0))
        selected_context = self._select_context_for_vae_attention(
            gen_context, allow_vae_condition=can_attend_vae_condition
        )
        past_key_values = selected_context["past_key_values"]
        kv_lens = selected_context["kv_lens"]
        ropes = selected_context["ropes"]
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
            ref_latents=[ref_latent],
        )
        generation_input = self._to_model_device(generation_input)

        # text cfg
        cfg_text_context = self._select_context_for_vae_attention(
            cfg_text_precontext, allow_vae_condition=can_attend_vae_condition
        )
        cfg_text_past_key_values = cfg_text_context["past_key_values"]
        kv_lens_cfg = cfg_text_context["kv_lens"]
        ropes_cfg = cfg_text_context["ropes"]
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )
        generation_input_cfg_text = self._to_model_device(generation_input_cfg_text)

        # img cfg
        cfg_img_context = self._select_context_for_vae_attention(
            cfg_img_precontext, allow_vae_condition=can_attend_vae_condition
        )
        cfg_img_past_key_values = cfg_img_context["past_key_values"]
        kv_lens_cfg = cfg_img_context["kv_lens"]
        ropes_cfg = cfg_img_context["ropes"]
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )
        generation_input_cfg_img = self._to_model_device(generation_input_cfg_img)

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text[
                "cfg_packed_position_ids"
            ],
            cfg_text_packed_query_indexes=generation_input_cfg_text[
                "cfg_packed_query_indexes"
            ],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text[
                "cfg_packed_key_value_indexes"
            ],
            cfg_img_packed_position_ids=generation_input_cfg_img[
                "cfg_packed_position_ids"
            ],
            cfg_img_packed_query_indexes=generation_input_cfg_img[
                "cfg_packed_query_indexes"
            ],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img[
                "cfg_packed_key_value_indexes"
            ],
            enable_taylorseer=enable_taylorseer,
        )

        image = self.decode_image(unpacked_latent[0], image_shape, return_tensor=False)
        return image

    def decode_image(self, latent, image_shape, return_tensor: bool = False):
        """
        Decode latent to image.

        Args:
            latent: VAE latent tensor
            image_shape: (H, W) tuple
            return_tensor: If True, return tensor (C, H, W) in [0, 1] (same as flow_grpo).
                          If False, return PIL Image.
        """
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(
            1,
            h,
            w,
            self.model.latent_patch_size,
            self.model.latent_patch_size,
            self.model.latent_channel,
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1,
            self.model.latent_channel,
            h * self.model.latent_patch_size,
            w * self.model.latent_patch_size,
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].float()  # (C, H, W), [0, 1]

        if return_tensor:
            # Return tensor directly (same as flow_grpo)
            return image
        else:
            # Return PIL Image for compatibility
            image_pil = image.permute(1, 2, 0) * 255  # (H, W, C)
            image_pil = Image.fromarray(image_pil.to(torch.uint8).cpu().numpy())
            return image_pil

    @torch.no_grad()
    def gen_text(
        self,
        gen_context,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
    ):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input = self.model.prepare_start_tokens(
            kv_lens, ropes, self.new_token_ids
        )
        generation_input = self._to_model_device(generation_input)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids["eos_token_id"],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:, 0])
        output = output.split("<|im_end|>")[0].split("<|im_start|>")[1]
        return output

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=True,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift: Optional[float] = None,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
        feedback_generated_image_to_context=True,
        feedback_generated_image_as_vae_condition=True,
        feedback_generated_image_as_vit_condition=True,
        num_generation_turns=1,
        generate_text_between_turns=False,
        max_turn_text_tokens=512,
        use_action_token_control=False,
        image_action_token="<|gen_image|>",
        max_auto_images=None,
        use_action_control_system_prompt=False,
        action_control_system_prompt=None,
        enable_size_tokens=False,
        enable_ref_image=False,
        text_can_attend_vae_condition=True,
        ref_target_can_attend_vae_condition=False,
        no_ref_target_can_attend_vae_condition=True,
        input_image_as_vae_condition: Optional[bool] = None,
        input_images: Optional[List[Union[Image.Image, torch.Tensor]]] = None,
        prefill_assistant_text: Optional[str] = None,
        inline_image_latents: bool = True,
    ) -> List[Union[str, Image.Image, Dict[str, Any]]]:
        diagnostics: Dict[str, Any] = {
            "mode": "legacy",
            "understanding_output": bool(understanding_output),
            "think": bool(think),
            "use_action_token_control": bool(use_action_token_control),
            "text_segments": 0,
            "image_outputs": 0,
            "output_items": 0,
            "action_token_hits": 0,
            "decode_steps": 0,
            "stop_reason": "completed",
            "action_token_hit_rate": 0.0,
            "image_generation_attempts": 0,
            "image_generation_success": 0,
            "image_feedback_applied": 0,
        }
        if timestep_shift is None:
            timestep_shift = float(getattr(self.model.config, "timestep_shift", 1.0))
        diagnostics["timestep_shift"] = float(timestep_shift)
        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        all_images: List[Union[Image.Image, torch.Tensor]] = list(input_images or [])

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if use_action_token_control and (use_action_control_system_prompt or think):
                system_prompt = (
                    action_control_system_prompt
                    or build_action_control_system_prompt(image_action_token)
                )
                if isinstance(system_prompt, str) and system_prompt.strip():
                    gen_context = self.update_context_text(system_prompt, gen_context)
                    cfg_img_context = self.update_context_text(
                        system_prompt, cfg_img_context
                    )

            if think and not use_action_token_control:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(
                    system_prompt, cfg_img_context
                )

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(
                        input_term, cfg_img_context
                    )

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    if input_images is None:
                        all_images.append(input_term)
                    if enable_size_tokens:
                        h, w = input_term.size[::-1]  # PIL .size is (W, H)
                        size_text = f"[Image size: ({h}, {w})]"
                        gen_context = self.update_context_text(size_text, gen_context)
                        cfg_img_context = self.update_context_text(
                            size_text, cfg_img_context
                        )
                    use_input_vae = (
                        not understanding_output
                        if input_image_as_vae_condition is None
                        else bool(input_image_as_vae_condition)
                    )
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=use_input_vae
                    )

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                output_list.append(gen_text)
                diagnostics["text_segments"] = 1 if isinstance(gen_text, str) else 0
                diagnostics["stop_reason"] = "understanding_output"

            else:
                if think and not use_action_token_control:
                    # ThinkMorph-style interleaved generation loop:
                    # gen_text → check for <image_start> → gen_image → loop
                    diagnostics["mode"] = "think_interleaved"
                    rounds = 0
                    max_rounds = max(int(num_generation_turns), 1)
                    while rounds < max_rounds:
                        gen_text = self.gen_text(
                            gen_context,
                            do_sample=do_sample,
                            temperature=text_temperature,
                            max_length=max_think_token_n,
                        )
                        output_list.append(gen_text)
                        gen_context = self.update_context_text(gen_text, gen_context)
                        if isinstance(gen_text, str):
                            diagnostics["text_segments"] += 1

                        if isinstance(gen_text, str) and "<image_start>" in gen_text:
                            diagnostics["image_generation_attempts"] += 1
                            img = self.gen_image(
                                image_shapes,
                                gen_context,
                                cfg_text_precontext=cfg_text_context,
                                cfg_img_precontext=cfg_img_context,
                                cfg_text_scale=cfg_text_scale,
                                cfg_img_scale=cfg_img_scale,
                                cfg_interval=cfg_interval,
                                timestep_shift=timestep_shift,
                                num_timesteps=num_timesteps,
                                cfg_renorm_min=cfg_renorm_min,
                                cfg_renorm_type=cfg_renorm_type,
                                enable_taylorseer=enable_taylorseer,
                            )
                            output_list.append(img)
                            diagnostics["image_outputs"] += 1
                            generated_image = img if isinstance(img, Image.Image) else None
                            diagnostics["image_generation_success"] += int(
                                generated_image is not None
                            )
                            if feedback_generated_image_to_context and generated_image is not None:
                                generated_image = self._to_pil_image(generated_image)
                                generated_image = self.vae_transform.resize_transform(
                                    generated_image
                                )
                                gen_context = self.update_context_image(
                                    generated_image,
                                    gen_context,
                                    vae=feedback_generated_image_as_vae_condition,
                                )
                                diagnostics["image_feedback_applied"] += 1
                            rounds += 1
                        else:
                            diagnostics["stop_reason"] = "no_image_start_in_text"
                            break

                elif use_action_token_control:
                    action_token_registered = self.new_token_ids.get(
                        "register_action_token", True
                    )
                    if action_token_registered:
                        # Existing path: single-token stop detection
                        action_token_id = self._resolve_image_action_token_id(
                            image_action_token
                        )
                        output_list.extend(
                            self._decode_interleaved_action_control(
                                gen_context=gen_context,
                                cfg_text_context=cfg_text_context,
                                cfg_img_context=cfg_img_context,
                                image_shapes=image_shapes,
                                max_text_tokens=max_turn_text_tokens,
                                do_sample=do_sample,
                                text_temperature=text_temperature,
                                image_action_token_id=action_token_id,
                                max_auto_images=max_auto_images,
                                cfg_text_scale=cfg_text_scale,
                                cfg_img_scale=cfg_img_scale,
                                cfg_interval=cfg_interval,
                                timestep_shift=timestep_shift,
                                num_timesteps=num_timesteps,
                                cfg_renorm_min=cfg_renorm_min,
                                cfg_renorm_type=cfg_renorm_type,
                                enable_taylorseer=enable_taylorseer,
                                feedback_generated_image_to_context=feedback_generated_image_to_context,
                                feedback_generated_image_as_vae_condition=feedback_generated_image_as_vae_condition,
                                feedback_generated_image_as_vit_condition=feedback_generated_image_as_vit_condition,
                                diagnostics=diagnostics,
                                enable_size_tokens=enable_size_tokens,
                                enable_ref_image=enable_ref_image,
                                text_can_attend_vae_condition=text_can_attend_vae_condition,
                                ref_target_can_attend_vae_condition=ref_target_can_attend_vae_condition,
                                no_ref_target_can_attend_vae_condition=no_ref_target_can_attend_vae_condition,
                                all_images=all_images,
                                prefill_assistant_text=prefill_assistant_text,
                                inline_image_latents=inline_image_latents,
                            )
                        )
                    else:
                        # Parse mode: generate text until EOS, then regex-parse
                        output_list.extend(
                            self._decode_interleaved_action_control_parse_mode(
                                gen_context=gen_context,
                                cfg_text_context=cfg_text_context,
                                cfg_img_context=cfg_img_context,
                                image_shapes=image_shapes,
                                max_text_tokens=max_turn_text_tokens,
                                do_sample=do_sample,
                                text_temperature=text_temperature,
                                image_action_token=image_action_token,
                                max_auto_images=max_auto_images,
                                cfg_text_scale=cfg_text_scale,
                                cfg_img_scale=cfg_img_scale,
                                cfg_interval=cfg_interval,
                                timestep_shift=timestep_shift,
                                num_timesteps=num_timesteps,
                                cfg_renorm_min=cfg_renorm_min,
                                cfg_renorm_type=cfg_renorm_type,
                                enable_taylorseer=enable_taylorseer,
                                feedback_generated_image_to_context=feedback_generated_image_to_context,
                                feedback_generated_image_as_vae_condition=feedback_generated_image_as_vae_condition,
                                feedback_generated_image_as_vit_condition=feedback_generated_image_as_vit_condition,
                                diagnostics=diagnostics,
                                enable_size_tokens=enable_size_tokens,
                                enable_ref_image=enable_ref_image,
                                text_can_attend_vae_condition=text_can_attend_vae_condition,
                                ref_target_can_attend_vae_condition=ref_target_can_attend_vae_condition,
                                no_ref_target_can_attend_vae_condition=no_ref_target_can_attend_vae_condition,
                                all_images=all_images,
                                prefill_assistant_text=prefill_assistant_text,
                            )
                        )
                else:
                    num_generation_turns = max(int(num_generation_turns), 1)
                    diagnostics["mode"] = "legacy_turns"
                    diagnostics["max_turns"] = int(num_generation_turns)
                    for turn_idx in range(num_generation_turns):
                        diagnostics["image_generation_attempts"] += 1
                        img = self.gen_image(
                            image_shapes,
                            gen_context,
                            cfg_text_precontext=cfg_text_context,
                            cfg_img_precontext=cfg_img_context,
                            cfg_text_scale=cfg_text_scale,
                            cfg_img_scale=cfg_img_scale,
                            cfg_interval=cfg_interval,
                            timestep_shift=timestep_shift,
                            num_timesteps=num_timesteps,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                            enable_taylorseer=enable_taylorseer,
                        )

                        output_list.append(img)
                        diagnostics["image_outputs"] += 1
                        generated_image = img if isinstance(img, Image.Image) else None
                        diagnostics["image_generation_success"] += int(
                            generated_image is not None
                        )
                        if feedback_generated_image_to_context and (
                            feedback_generated_image_as_vae_condition
                            or feedback_generated_image_as_vit_condition
                        ):
                            if generated_image is not None:
                                generated_image = self._to_pil_image(generated_image)
                                generated_image = self.vae_transform.resize_transform(
                                    generated_image
                                )
                                gen_context = self.update_context_image(
                                    generated_image,
                                    gen_context,
                                    vae=feedback_generated_image_as_vae_condition,
                                    vit=feedback_generated_image_as_vit_condition,
                                )
                                diagnostics["image_feedback_applied"] += 1
                                image_shapes = generated_image.size[::-1]
                                cfg_text_context = deepcopy(gen_context)
                                cfg_img_context = deepcopy(gen_context)

                        if (
                            generate_text_between_turns
                            and turn_idx < num_generation_turns - 1
                        ):
                            inter_text = self.gen_text(
                                gen_context,
                                do_sample=do_sample,
                                temperature=text_temperature,
                                max_length=max_turn_text_tokens,
                            )
                            output_list.append(inter_text)
                            if isinstance(inter_text, str):
                                diagnostics["text_segments"] += 1
                            gen_context = self.update_context_text(
                                inter_text, gen_context
                            )
                            cfg_img_context = self.update_context_text(
                                inter_text, cfg_img_context
                            )
                            cfg_text_context = deepcopy(gen_context)

        # Fallback counts for branches that did not set them explicitly.
        if diagnostics.get("mode") != "action_token":
            image_outputs = 0
            text_segments = 0
            for item in output_list:
                if isinstance(item, str):
                    text_segments += 1
                elif isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        text_segments += 1
                    if item.get("image") is not None:
                        image_outputs += 1
                elif isinstance(item, Image.Image) or torch.is_tensor(item):
                    image_outputs += 1
            diagnostics["text_segments"] = text_segments
            diagnostics["image_outputs"] = image_outputs

        diagnostics["output_items"] = len(output_list)
        self.last_interleave_diagnostics = diagnostics
        return output_list

    def __call__(
        self, image: Optional[Image.Image] = None, text: Optional[str] = None, **kargs
    ) -> Dict[str, Any]:
        output_dict = {"image": None, "text": None}

        if image is None and text is None:
            print("Please provide at least one input: either an image or text.")
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict["image"] = i
            elif isinstance(i, str):
                output_dict["text"] = i
            elif isinstance(i, dict):
                output_dict.update(i)
        return output_dict
