import json
import os
import re
from typing import Dict, Iterable

import torch
from loguru import logger

from data.collator import BagelCollator
from data.iterable.adaptive_image_sizing import (
    adapt_image_sizes_to_fit_budget,
)
from data.iterable.multimodal_iterable_dataset import (
    MultiModalIterableDataset,
)
from data.registry import register_dataset
import torch.distributed as dist
from utils.train_utils import TrainUtilities


@register_dataset("bagel_iterable")
class BagelIterableDataset(MultiModalIterableDataset):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._missing_text_dropped_count = 0
        self._missing_reasoning_dropped_count = 0
        self.use_action_control_system_prompt = bool(
            getattr(config, "use_action_control_system_prompt", False)
        )
        self.action_control_system_prompt = getattr(
            config, "action_control_system_prompt", None
        )
        self.wrap_assistant_reasoning_with_think = bool(
            getattr(config, "wrap_assistant_reasoning_with_think", False)
        )
        # Legacy flag removed - action-token text format is always active.

    def _processor_extra_kwargs(self) -> Dict:
        extra_kwargs = getattr(self.processor_config, "extra_kwargs", None)
        return dict(extra_kwargs or {})

    def _wrap_reasoning_with_think(self, text: str) -> str:
        value = text.strip()
        if not value:
            return value
        if "<think>" in value and "</think>" in value:
            return value
        return f"<think>{value}</think>"

    def _prepend_system_prompt_if_needed(self, messages):
        if not self.use_action_control_system_prompt:
            return messages
        if (
            not self.action_control_system_prompt
            or not self.action_control_system_prompt.strip()
        ):
            return messages
        if (
            messages
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "system"
        ):
            content = messages[0].get("content", [])
            if (
                isinstance(content, list)
                and len(content) == 1
                and isinstance(content[0], dict)
                and content[0].get("type") == "text"
                and str(content[0].get("text", "")).strip()
                == self.action_control_system_prompt.strip()
            ):
                return messages
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.action_control_system_prompt}
                ],
            },
            *messages,
        ]

    def _iter_text_chunks(self, content) -> Iterable[str]:
        if content is None:
            return []
        if isinstance(content, str):
            return [content]
        if isinstance(content, dict):
            if content.get("type") == "text":
                return [content.get("text", "")]
            return []
        if isinstance(content, list):
            chunks = []
            for item in content:
                chunks.extend(self._iter_text_chunks(item))
            return chunks
        return []

    def _has_text_for_role(self, messages, role: str) -> bool:
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            if message.get("role") != role:
                continue
            content = message.get("content")
            for text in self._iter_text_chunks(content):
                if isinstance(text, str) and text.strip():
                    return True
        return False

    def _has_content_for_role(self, messages, role: str) -> bool:
        """Check if a role has any content (text or image)."""
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            if message.get("role") != role:
                continue
            content = message.get("content")
            if not content:
                continue
            if isinstance(content, str) and content.strip():
                return True
            if isinstance(content, list) and len(content) > 0:
                return True
            if isinstance(content, dict) and content.get("type"):
                return True
        return False

    def _has_required_reasoning_step_zero_and_answer(self, row: Dict) -> bool:
        thought = row.get("reasoning_thought_0")
        image = row.get("reasoning_image_0")
        thought_ok = isinstance(thought, str) and thought.strip() != ""
        image_ok = isinstance(image, str) and image.strip() != ""
        answer_ok = self._get_answer_text(row) is not None
        if thought_ok and image_ok and answer_ok:
            return True
        _is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        if _is_rank0:
            missing = []
            if not thought_ok:
                missing.append("reasoning_thought_0")
            if not image_ok:
                missing.append("reasoning_image_0")
            if not answer_ok:
                missing.append("gt_answer/answer")
            self._missing_reasoning_dropped_count += 1
            logger.warning(
                "Filtering sample missing required fields: "
                f"{','.join(missing)}. "
                f"Total dropped so far: {self._missing_reasoning_dropped_count}"
            )
        return False

    def _check_missing_text(self, messages) -> bool:
        has_user_text = self._has_text_for_role(messages, "user")
        has_assistant_content = self._has_content_for_role(messages, "assistant")
        if has_user_text and has_assistant_content:
            return True
        if self.config.filter_missing_text:
            missing = []
            if not has_user_text:
                missing.append("user")
            if not has_assistant_content:
                missing.append("assistant")
            _is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
            if _is_rank0:
                self._missing_text_dropped_count += 1
                logger.warning(
                    f"Filtering sample missing content for roles: "
                    f"{','.join(missing)}. "
                    f"Total dropped so far: {self._missing_text_dropped_count}"
                )
            return False
        return True

    def _parse_image_list_column(self, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return []
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except json.JSONDecodeError:
                pass
            return [candidate]
        return []

    def _join_media_root(self, root: str | None, media_path: str) -> str:
        media_path = str(media_path).strip()
        if not root:
            return media_path
        if os.path.isabs(media_path):  # absolute paths are used as given
            return media_path
        return os.path.join(root, media_path)

    def _collect_reasoning_indices(self, row: Dict) -> list[int]:
        indices = set()
        for key in row.keys():
            match = re.match(r"reasoning_(?:thought|image|ref_image)_(\d+)$", key)
            if match:
                indices.add(int(match.group(1)))
        return sorted(indices)

    def _get_thought_text(self, row: Dict, idx: int) -> str:
        """Extract and optionally wrap a reasoning thought."""
        thought = row.get(f"reasoning_thought_{idx}")
        if thought is None:
            return ""
        if isinstance(thought, str):
            text = thought.strip()
        else:
            text = str(thought).strip()
        if not text:
            return ""
        if self.wrap_assistant_reasoning_with_think:
            text = self._wrap_reasoning_with_think(text)
        return text

    def _get_answer_text(self, row: Dict) -> str | None:
        """Return the answer string, trying ``gt_answer`` then ``answer``."""
        for key in ("gt_answer", "answer"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _format_question_with_options(self, row: Dict) -> str:
        question_value = row.get("question", "")
        if question_value is None:
            question = ""
        elif isinstance(question_value, str):
            question = question_value.strip()
        else:
            question = str(question_value).strip()

        options_value = row.get("options")
        if options_value is None:
            return question
        if isinstance(options_value, str):
            options = options_value.strip()
        else:
            options = str(options_value).strip()
        if not options:
            return question
        if re.search(r"\boptions\s*:", question, flags=re.IGNORECASE):
            return question
        if not question:
            return f"Options: {options}"
        return f"{question}\nOptions: {options}"

    def _row_images_root(self, row: Dict) -> str | None:
        """Per-row images root (set by folder-based loading), else config."""
        return row.get("_images_root") or self.config.jsonl_images_root

    def _row_reasoning_images_root(self, row: Dict) -> str | None:
        return row.get("_images_root") or self.config.jsonl_reasoning_images_root

    def _build_reasoning_jsonl_messages(self, row: Dict) -> Dict:
        question = self._format_question_with_options(row)

        images_root = self._row_images_root(row)
        reasoning_images_root = self._row_reasoning_images_root(row)

        user_content = []
        images = self._parse_image_list_column(row.get("images"))
        for image in images:
            image_path = self._join_media_root(images_root, image)
            user_content.append({"type": "image_url", "image_url": {"url": image_path}})
        if question:
            user_content.append({"type": "text", "text": question})

        # Build unified image numbering: map "input_K" and "reasoning_image_K"
        # to sequential indices (0, 1, 2, ...) across all images in order.
        # This is used for ref_image resolution.
        unified_name_to_idx: Dict[str, int] = {}
        unified_counter = 0
        # User images first
        for k in range(len(images)):
            unified_name_to_idx[f"input_{k}"] = unified_counter
            unified_counter += 1

        assistant_content = self._build_action_token_assistant_content(
            row,
            unified_name_to_idx,
            unified_counter,
            reasoning_images_root,
        )

        messages = []
        if user_content:
            messages.append({"role": "user", "content": user_content})
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})
        messages = self._prepend_system_prompt_if_needed(messages)

        # Build ref_image_mapping: parallel to the images list, maps each
        # image to its resolved unified ref index (or None).
        # Images list order: user images first, then assistant images in order.
        ref_image_mapping = [None] * len(images)  # user images have no ref
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for content in msg.get("content", []):
                if content.get("type") == "image_url":
                    ref_name = content.pop("ref_image", None)
                    if isinstance(ref_name, str) and ref_name.strip():
                        ref_idx = unified_name_to_idx.get(ref_name.strip())
                        ref_image_mapping.append(ref_idx)
                    else:
                        ref_image_mapping.append(None)

        return {"messages": messages, "ref_image_mapping": ref_image_mapping}

    def _build_action_token_assistant_content(
        self,
        row: Dict,
        unified_name_to_idx: Dict,
        unified_counter: int,
        reasoning_images_root: str | None = None,
    ) -> list:
        """Assistant content for action-token-supervised reasoning datasets.

        Preserve ThinkMorph-style text grouping and answer formatting while
        representing images as actual assistant image items rather than legacy
        ``<image_start>/<image_end>`` text markers.
        """
        assistant_content: list = []
        pending_text_parts: list[str] = []

        for idx in self._collect_reasoning_indices(row):
            thought_text = self._get_thought_text(row, idx)
            if thought_text:
                pending_text_parts.append(thought_text)

            reasoning_image = row.get(f"reasoning_image_{idx}")
            if isinstance(reasoning_image, str) and reasoning_image.strip():
                merged = "".join(pending_text_parts)
                if merged:
                    assistant_content.append({"type": "text", "text": merged})
                pending_text_parts = []

                reasoning_path = self._join_media_root(
                    reasoning_images_root,
                    reasoning_image,
                )
                content_item = {
                    "type": "image_url",
                    "image_url": {"url": reasoning_path},
                }
                reasoning_ref = row.get(f"reasoning_ref_image_{idx}")
                if isinstance(reasoning_ref, str) and reasoning_ref.strip():
                    content_item["ref_image"] = reasoning_ref.strip()
                assistant_content.append(content_item)
                unified_name_to_idx[f"reasoning_image_{idx}"] = unified_counter
                unified_counter += 1

        final_answer = self._get_answer_text(row)
        if final_answer:
            pending_text_parts.append(f"<answer>{final_answer}</answer>")

        merged = "".join(pending_text_parts)
        if merged:
            assistant_content.append({"type": "text", "text": merged})

        return assistant_content

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        # TODO Write a protocol for vision openai input
        images_list = []
        ref_image_mapping = None
        sample_id = None
        if isinstance(data, dict):
            sid_val = data.get("id")
            if sid_val is not None:
                sample_id = str(sid_val)
        if (
            self.config.dataset_format == "jsonl"
            and self.config.jsonl_mode == "reasoning_columns"
        ):
            if getattr(
                self.config, "require_reasoning_image_0_and_answer", False
            ) and not self._has_required_reasoning_step_zero_and_answer(data):
                return None
            data = self._build_reasoning_jsonl_messages(data)
            ref_image_mapping = data.get("ref_image_mapping")
        messages = self._prepend_system_prompt_if_needed(data["messages"])
        if not self._check_missing_text(messages):
            return None
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        images = [
            self.load_image(image, data_folder=data_folder) for image in images_list
        ]
        adaptive_enabled = bool(
            getattr(self.config, "adaptive_image_size_to_fit_budget", True)
        ) and len(images) > 0
        budget = None
        if adaptive_enabled:
            budget = (
                getattr(self.config, "max_num_tokens_per_sample", None)
                or getattr(self.config, "max_num_tokens", None)
            )
        adapter_result = None
        if adaptive_enabled and budget:
            # Estimate-based shrink of reasoning (assistant) images only.
            # User images are kept at full size. If the floor still doesn't
            # fit the budget, the adapter signals exceeds_budget=True and
            # we mark the sample so it bypasses the iterable's drop check.
            adapter_result = adapt_image_sizes_to_fit_budget(
                hf_messages=hf_messages,
                images=images,
                processor=self.processor,
                budget=int(budget),
                safety_margin=float(
                    getattr(
                        self.config,
                        "adaptive_image_size_safety_margin",
                        1.05,
                    )
                ),
                floor_size=int(
                    getattr(self.config, "adaptive_image_size_floor", 512)
                ),
                sample_id=sample_id,
            )
        proc_images = images if len(images) > 0 else None
        inputs = self.processor.process(
            images=proc_images,
            hf_messages=hf_messages,
            ref_image_mapping=ref_image_mapping,
        )
        if (
            inputs is not None
            and adapter_result is not None
            and adapter_result.get("exceeds_budget")
        ):
            # Adapter floored the assistant images but estimate still exceeded
            # budget. Mark the sample so the downstream iterable does NOT drop it.
            inputs["_adaptive_keep_overlong"] = True
        return inputs

    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        messages = self._prepend_system_prompt_if_needed(data["messages"])
        if not self._check_missing_text(messages):
            return None
        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if isinstance(data["image"], list):
            images = data["image"]
        else:
            images = [data["image"]]
        ref_image_mapping = data.get("ref_image_mapping")
        inputs = self.processor.process(
            images=images,
            hf_messages=hf_messages,
            ref_image_mapping=ref_image_mapping,
        )
        return inputs

    def get_collator(self):
        return BagelCollator(self.processor)
