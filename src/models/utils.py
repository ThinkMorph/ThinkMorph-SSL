from loguru import logger
from transformers import PretrainedConfig

from utils import TrainUtilities

VALID_CONFIG_TYPE = {
    "bagel",
}


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        if config.model_type not in VALID_CONFIG_TYPE:
            logger.warning(
                f"Only support config type of {VALID_CONFIG_TYPE}, but got {config.model_type}. MFU will not be counted."
            )

        self.estimate_func = {
            "bagel": self._estimate_qwen2_flops,
        }
        if config.model_type in [
            "llava_onevision",
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen2_5_omni",
            "qwen2_5_omni_thinker",
            "qwen3_omni_moe",
            "qwen3_omni_moe_thinker",
        ]:
            self.config = config.text_config
            self.config.model_type = config.model_type
        elif config.model_type == "bagel":
            self.config = config.llm_config
            self.config.model_type = config.model_type
        else:
            self.config = config

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time):
        return 0

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        config = self.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        num_attention_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size

        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (
            q_size + k_size + v_size + num_attention_heads * head_dim
        )
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = (
            12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
        )

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens, delta_time):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the
                current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        # Flatten batch_seqlens if it contains nested lists (common for vision models)
        if isinstance(batch_seqlens, (list, tuple)):
            flat_seqlens = []
            for item in batch_seqlens:
                if isinstance(item, (list, tuple)):
                    flat_seqlens.extend(item)
                else:
                    flat_seqlens.append(item)
            batch_seqlens = flat_seqlens
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(
            self.config.model_type, self._estimate_unknown_flops
        )
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time)
        promised_flops = TrainUtilities.get_device_flops()
        return estimated_flops, promised_flops


def setup_flops_counter(config: PretrainedConfig) -> FlopsCounter:
    """
    Setup the FlopsCounter based on the provided configuration.

    Args:
        config (PretrainedConfig): The configuration object for the model.

    Returns:
        FlopsCounter: An instance of FlopsCounter initialized with the given configuration.
    """
    global flops_counter
    flops_counter = FlopsCounter(config)
