"""Per-sample adaptive image sizing to keep packed token counts within budget.

The purpose is to ensure no training sample is dropped due to exceeding
``max_num_tokens_per_sample``. For samples whose default-resolution token cost
would overflow the budget, **assistant (reasoning) images** are shrunk
in-place to the largest size that fits, BEFORE they are handed to the
processor. User (input) images are LEFT AT ORIGINAL SIZE -- they're the
problem context and shouldn't lose fidelity.

A configurable floor size (default 512px) caps how aggressively reasoning
images get shrunk. If even at the floor a sample still exceeds the budget,
we ADMIT IT ANYWAY (return ``exceeds_budget=True`` so the iterable can
bypass its drop check) rather than discard a hard-to-fit sample.

The estimator deliberately models the WORST CASE (no condition dropout fires),
so the actual processed length is always less than or equal to the estimate.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from PIL import Image
from loguru import logger


# Try sizes from most-permissive to most-restrictive when shrinking reasoning
# images. Sizes >= the floor parameter are kept; below the floor we stop
# shrinking and admit the sample as-is.
DEFAULT_VAE_CANDIDATE_SIZES: Tuple[int, ...] = (
    1024, 896, 768, 640, 512,
)


def _make_div(value: float, stride: int) -> int:
    return max(stride, int(round(value / stride) * stride))


def _resized_dims(
    orig_w: int,
    orig_h: int,
    max_size: int,
    min_size: int,
    stride: int,
    max_pixels: int,
) -> Tuple[int, int]:
    """Replicate ``MaxLongEdgeMinShortEdgeResize.forward`` without loading bytes."""
    if orig_w <= 0 or orig_h <= 0:
        return stride, stride
    scale = min(max_size / max(orig_w, orig_h), 1.0)
    scale = max(scale, min_size / min(orig_w, orig_h))
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    new_w = _make_div(new_w, stride)
    new_h = _make_div(new_h, stride)
    if new_w * new_h > max_pixels:
        s = max_pixels / (new_w * new_h)
        new_w = round(new_w * s)
        new_h = round(new_h * s)
        new_w = _make_div(new_w, stride)
        new_h = _make_div(new_h, stride)
    if max(new_w, new_h) > max_size:
        s = max_size / max(new_w, new_h)
        new_w = round(new_w * s)
        new_h = round(new_h * s)
        new_w = _make_div(new_w, stride)
        new_h = _make_div(new_h, stride)
    return new_w, new_h


# Image transform default max_pixels (used internally by ImageTransform when
# the caller doesn't pass it). Mirrors transforms.ImageTransform default.
_VAE_TRANSFORM_DEFAULT_MAX_PIXELS = 14 * 14 * 9 * 1024  # 1,806,336


def _vae_tokens_for(orig_w: int, orig_h: int, vae_max: int, processor) -> int:
    new_w, new_h = _resized_dims(
        orig_w,
        orig_h,
        max_size=vae_max,
        min_size=processor.vae_min_image_size,
        stride=processor.vae_image_stride,
        max_pixels=_VAE_TRANSFORM_DEFAULT_MAX_PIXELS,
    )
    return (new_w // processor.vae_image_downsample) * (
        new_h // processor.vae_image_downsample
    )


def _vit_tokens_for(orig_w: int, orig_h: int, vit_max: int, processor) -> int:
    new_w, new_h = _resized_dims(
        orig_w,
        orig_h,
        max_size=vit_max,
        min_size=processor.vit_min_image_size,
        stride=processor.vit_image_stride,
        max_pixels=processor.vit_max_pixels,
    )
    return (new_w // processor.vit_patch_size) * (new_h // processor.vit_patch_size)


# Per-image fixed overhead (special tokens + image-size text label).
# Conservative upper bound to avoid under-counting:
#   - "[Image size: (HHHH, WWWW)]" tokenizes to ~12 tokens
#   - 2 special wrappers (start/end of image) per VAE/ViT block
#   - + 1-2 action token bytes for assistant images
_USER_IMAGE_TEXT_OVERHEAD = 16   # size_text + 2 specials + small chat slack
_ASSIST_IMAGE_TEXT_OVERHEAD = 22  # action suffix + size_text + 2 specials + slack


def _compute_text_skeleton(
    hf_messages: Sequence[dict],
    images: Sequence[Image.Image],
    processor,
) -> Tuple[int, List[Tuple[str, Tuple[int, int]]], bool]:
    """One-shot pass over messages: count text tokens + capture per-image
    (role, original_size) tuples. The text count, image roles, and original
    sizes do NOT depend on chosen image sizes, so this is computed once and
    reused across candidate sizes by the adapter.

    Returns (text_tokens, [(role, (w, h)), ...], has_assistant_image).
    """
    tok = processor.processor
    per_message_overhead = 8
    text_tokens = 0
    image_meta: List[Tuple[str, Tuple[int, int]]] = []
    has_assistant_image = False
    image_idx = 0
    for message in hf_messages:
        role = message["role"]
        contents = message.get("content") or []
        text_tokens += per_message_overhead
        for content in contents:
            ctype = content.get("type")
            if ctype == "text":
                txt = content.get("text") or ""
                if txt:
                    text_tokens += len(tok.encode(txt, add_special_tokens=False))
            elif ctype == "image":
                if role == "assistant":
                    has_assistant_image = True
                    text_tokens += _ASSIST_IMAGE_TEXT_OVERHEAD
                else:
                    text_tokens += _USER_IMAGE_TEXT_OVERHEAD
                w, h = images[image_idx].size if image_idx < len(images) else (0, 0)
                image_meta.append((role, (w, h)))
                image_idx += 1
    return text_tokens, image_meta, has_assistant_image


def estimate_sample_tokens(
    hf_messages: Sequence[dict],
    images: Sequence[Image.Image],
    processor,
    *,
    vae_max_override: int | None = None,
    vae_max_for_assistant: int | None = None,
    vae_max_for_user: int | None = None,
    text_skeleton: Tuple[int, List[Tuple[str, Tuple[int, int]]], bool] | None = None,
) -> Tuple[int, dict]:
    """Estimate WORST-CASE packed token count for one sample.

    Returns (total_estimate, debug_dict).
    Worst case == every conditional pass kept (no vae/vit cond dropout).
    Per-role VAE max overrides allow simulating "shrink assistant images only
    while leaving user images at their default size".

    For repeated calls on the same sample with different vae_max sizes (the
    adapter's candidate-size loop), pass a precomputed ``text_skeleton`` from
    ``_compute_text_skeleton`` to skip retokenizing text.
    """
    vae_max_default = vae_max_override or processor.vae_max_image_size
    vae_max_assistant = vae_max_for_assistant or vae_max_default
    vae_max_user = vae_max_for_user or vae_max_default
    vit_max = processor.vit_max_image_size

    if text_skeleton is None:
        text_tokens, image_meta, has_assistant_image = _compute_text_skeleton(
            hf_messages, images, processor
        )
    else:
        text_tokens, image_meta, has_assistant_image = text_skeleton

    use_user_vae_cond = (
        processor.user_image_as_vae_condition
        if processor.user_image_as_vae_condition is not None
        else has_assistant_image
    )

    image_token_breakdown = []
    for role, (w, h) in image_meta:
        if role == "assistant":
            vae_target = _vae_tokens_for(w, h, vae_max_assistant, processor)
            vae_cond = vae_target if processor.assistant_image_as_vae_condition else 0
            vit_cond = (
                _vit_tokens_for(w, h, vit_max, processor)
                if processor.assistant_image_as_vit_condition
                else 0
            )
            image_token_breakdown.append(
                {
                    "role": "assistant",
                    "orig": (w, h),
                    "vae_target": vae_target,
                    "vae_cond": vae_cond,
                    "vit_cond": vit_cond,
                }
            )
        else:
            vae_cond = (
                _vae_tokens_for(w, h, vae_max_user, processor)
                if use_user_vae_cond
                else 0
            )
            vit_tok = _vit_tokens_for(w, h, vit_max, processor)
            image_token_breakdown.append(
                {
                    "role": "user",
                    "orig": (w, h),
                    "vae_cond": vae_cond,
                    "vit": vit_tok,
                }
            )

    image_tokens = sum(
        b.get("vae_target", 0)
        + b.get("vae_cond", 0)
        + b.get("vit_cond", 0)
        + b.get("vit", 0)
        for b in image_token_breakdown
    )
    total = text_tokens + image_tokens
    return total, {
        "text_tokens": text_tokens,
        "image_tokens": image_tokens,
        "vae_max_assistant": vae_max_assistant,
        "vae_max_user": vae_max_user,
        "image_breakdown": image_token_breakdown,
    }


def _identify_assistant_image_indices(hf_messages: Sequence[dict]) -> List[int]:
    """Return list of image indices (0-based across all images in the sample)
    that belong to assistant turns (i.e., reasoning images)."""
    assistant_indices: List[int] = []
    image_idx = 0
    for message in hf_messages:
        role = message.get("role")
        for content in message.get("content") or []:
            if content.get("type") == "image":
                if role == "assistant":
                    assistant_indices.append(image_idx)
                image_idx += 1
    return assistant_indices


def adapt_image_sizes_to_fit_budget(
    hf_messages: Sequence[dict],
    images: List[Image.Image],
    processor,
    budget: int,
    safety_margin: float = 1.05,
    floor_size: int = 512,
    candidate_sizes: Sequence[int] = DEFAULT_VAE_CANDIDATE_SIZES,
    sample_id: str | None = None,
) -> dict:
    """Shrink ASSISTANT (reasoning) images in-place if estimate exceeds budget.

    Strategy:
      1) Estimate at default sizes; if fits, do nothing.
      2) Else try shrinking only assistant images down through candidate sizes
         that are >= ``floor_size``. Pick the largest size that fits.
      3) If even at ``floor_size`` the estimate still exceeds budget, apply
         floor_size to assistant images and return ``exceeds_budget=True``.
         Caller should admit the sample anyway (skip drop).

    User (input) images are NEVER shrunk -- they are the problem context.

    Returns a dict:
      ``shrunk`` (bool), ``chosen_vae_max`` (int, applied to assistant imgs),
      ``before_estimate``, ``after_estimate``, ``exceeds_budget`` (bool).
    """
    # Compute text skeleton ONCE; reuse across the candidate-size loop so
    # we never retokenize the (unchanged) text.
    skel = _compute_text_skeleton(hf_messages, images, processor)
    before_estimate, _ = estimate_sample_tokens(
        hf_messages, images, processor, text_skeleton=skel
    )
    effective_budget = int(budget / max(safety_margin, 1.0))
    default_max = processor.vae_max_image_size

    if before_estimate <= effective_budget:
        return {
            "shrunk": False,
            "chosen_vae_max": default_max,
            "before_estimate": before_estimate,
            "after_estimate": before_estimate,
            "exceeds_budget": False,
        }

    assistant_indices = _identify_assistant_image_indices(hf_messages)
    if not assistant_indices:
        # No assistant images to shrink; nothing we can do without touching
        # input images. Admit as-is.
        logger.warning(
            f"[adaptive_image_size] sample{'='+sample_id if sample_id else ''} "
            f"est={before_estimate} > budget={budget} but no assistant images "
            f"to shrink; admitting as-is (sample kept)."
        )
        return {
            "shrunk": False,
            "chosen_vae_max": default_max,
            "before_estimate": before_estimate,
            "after_estimate": before_estimate,
            "exceeds_budget": True,
        }

    # Candidates strictly below default and >= floor (largest first).
    descending = sorted(
        {s for s in candidate_sizes if s < default_max and s >= floor_size},
        reverse=True,
    )
    chosen: int | None = None
    final_estimate = before_estimate
    for vae_max in descending:
        est, _ = estimate_sample_tokens(
            hf_messages,
            images,
            processor,
            vae_max_for_assistant=vae_max,
            vae_max_for_user=default_max,
            text_skeleton=skel,
        )
        if est <= effective_budget:
            chosen = vae_max
            final_estimate = est
            break

    exceeds_budget = False
    if chosen is None:
        # Floor the assistant images even if it still exceeds; admit anyway.
        chosen = floor_size
        final_estimate, _ = estimate_sample_tokens(
            hf_messages,
            images,
            processor,
            vae_max_for_assistant=chosen,
            vae_max_for_user=default_max,
            text_skeleton=skel,
        )
        exceeds_budget = final_estimate > effective_budget

    # Apply: shrink ONLY assistant images in-place to longest side <= chosen.
    for idx in assistant_indices:
        if idx >= len(images):
            continue
        img = images[idx]
        if img is None:
            continue
        if max(img.size) > chosen:
            img.thumbnail((chosen, chosen), Image.LANCZOS)

    log_fn = logger.error if exceeds_budget else logger.warning
    log_fn(
        f"[adaptive_image_size] sample{'='+sample_id if sample_id else ''} "
        f"reasoning_image vae_max {default_max}->{chosen} "
        f"(est tokens {before_estimate}->{final_estimate}, budget={budget}, "
        f"exceeds_budget={exceeds_budget})"
    )

    return {
        "shrunk": True,
        "chosen_vae_max": chosen,
        "before_estimate": before_estimate,
        "after_estimate": final_estimate,
        "exceeds_budget": exceeds_budget,
    }
