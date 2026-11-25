from types import SimpleNamespace

from PIL import Image

from models.bagel.inferencer import (
    InterleaveInferencer,
    build_action_control_system_prompt,
)


class _FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.mapping[token_id] for token_id in token_ids)


def _make_inferencer(token_mapping, sampled_tokens):
    inferencer = InterleaveInferencer.__new__(InterleaveInferencer)
    inferencer.tokenizer = _FakeTokenizer(token_mapping)
    inferencer.new_token_ids = {"eos_token_id": 0, "bos_token_id": 99}
    inferencer.vae_transform = SimpleNamespace(resize_transform=lambda image: image)
    inferencer.model = SimpleNamespace()
    inferencer._sampled_tokens = iter(sampled_tokens)
    inferencer._event_log = []
    return inferencer


def test_parse_mode_regenerates_post_image_text_after_image_feedback():
    inferencer = _make_inferencer(
        token_mapping={
            1: "<think>thought 0</think>",
            2: "<gen_image>(512, 512)",
            3: "<think>thought 1</think>",
            4: "<answer>done</answer>",
        },
        sampled_tokens=[1, 2, 3, 4, 0],
    )

    def init_decode_state(_gen_context):
        return {"kv_len": 0, "rope": 0, "past_key_values": None}

    def forward_decode_text_token(_decode_state, token_id, **_kwargs):
        inferencer._event_log.append(f"sample:{token_id}")
        return token_id

    def sample_next_token_id(_packed_query, **_kwargs):
        return next(inferencer._sampled_tokens)

    def forward_single_text_token(_decode_state, token_id, **_kwargs):
        inferencer._event_log.append(f"forward:{token_id}")
        return token_id

    def sync_decode_state_to_context(_decode_state, gen_context):
        return gen_context

    def update_context_text(text, gen_context):
        inferencer._event_log.append(f"text:{text}")
        return gen_context

    def gen_image(*_args, **_kwargs):
        inferencer._event_log.append("image")
        return {"image": Image.new("RGB", (8, 8), "white")}

    def update_context_image(image, gen_context, **_kwargs):
        inferencer._event_log.append(f"image_feedback:{image.size}")
        return gen_context

    inferencer._init_single_decode_state = init_decode_state
    inferencer._forward_decode_text_token = forward_decode_text_token
    inferencer._sample_next_token_id = sample_next_token_id
    inferencer._forward_single_text_token = forward_single_text_token
    inferencer._sync_decode_state_to_context = sync_decode_state_to_context
    inferencer.update_context_text = update_context_text
    inferencer.gen_image = gen_image
    inferencer.update_context_image = update_context_image
    inferencer._to_pil_image = lambda image: image
    inferencer._decode_text_until_stop_with_generate_text = lambda *args, **kwargs: (
        "<answer>stale tail</answer>",
        0,
        1,
    )

    outputs = inferencer._decode_interleaved_action_control_parse_mode(
        gen_context={"kv_lens": [0], "ropes": [0], "past_key_values": None},
        cfg_text_context={},
        cfg_img_context={},
        image_shapes=(512, 512),
        max_text_tokens=16,
        do_sample=False,
        text_temperature=1.0,
        image_action_token="<gen_image>",
        max_auto_images=1,
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
        cfg_interval=(0.0, 1.0),
        timestep_shift=1.0,
        num_timesteps=1,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
        enable_taylorseer=False,
        feedback_generated_image_to_context=True,
        feedback_generated_image_as_vae_condition=True,
        feedback_generated_image_as_vit_condition=True,
        diagnostics={},
        enable_size_tokens=True,
        text_can_attend_vae_condition=True,
    )

    assert outputs[0] == "<think>thought 0</think>"
    assert outputs[2] == "<think>thought 1</think><answer>done</answer>"
    assert all(
        "stale tail" not in output for output in outputs if isinstance(output, str)
    )
    assert "text:<think>thought 0</think><gen_image>(512, 512)" in inferencer._event_log
    assert "image" in inferencer._event_log
    assert "sample:3" in inferencer._event_log
    assert "sample:4" in inferencer._event_log
    assert inferencer._event_log.index("image") < inferencer._event_log.index(
        "sample:3"
    )


def test_action_control_prompt_uses_configured_image_action_token():
    prompt = build_action_control_system_prompt("<gen_image>")

    assert "<gen_image>" in prompt
    assert "<|gen_image|>" not in prompt
