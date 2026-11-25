"""Model registration utilities.

Previously in lmms_engine.mapping_func. Import from here instead.
"""
from typing import Literal

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMaskedLM,
    PretrainedConfig,
)
from transformers.modeling_utils import PreTrainedModel

AUTO_REGISTER_MODEL_MAPPING = {
    "causal_lm": AutoModelForCausalLM,
    "masked_lm": AutoModelForMaskedLM,
    "image_text_to_text": AutoModelForImageTextToText,
    "general": AutoModel,
}


def register_model(
    model_type: str,
    model_config: PretrainedConfig,
    model_class: PreTrainedModel,
    model_general_type: Literal[
        "causal_lm", "masked_lm", "image_text_to_text", "general"
    ] = "causal_lm",
):
    AutoConfig.register(model_type, model_config, exist_ok=True)
    AUTO_REGISTER_MODEL_MAPPING[model_general_type].register(model_config, model_class)


def create_model_from_pretrained(load_from_pretrained_path):
    config = AutoConfig.from_pretrained(load_from_pretrained_path)

    if type(config) in AutoModelForCausalLM._model_mapping.keys():
        model_class = AutoModelForCausalLM
    elif type(config) in AutoModelForImageTextToText._model_mapping.keys():
        model_class = AutoModelForImageTextToText
    elif type(config) in AutoModelForMaskedLM._model_mapping.keys():
        model_class = AutoModelForMaskedLM
    elif type(config) in AutoModel._model_mapping.keys():
        model_class = AutoModel
    else:
        raise ValueError(f"Model {load_from_pretrained_path} is not supported.")
    return model_class


def create_model_from_config(model_type, config):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if model_type in CONFIG_MAPPING:
        config_class = CONFIG_MAPPING[model_type]
        m_config = config_class(**config)
        if type(m_config) in AutoModelForCausalLM._model_mapping.keys():
            model_class = AutoModelForCausalLM
        elif type(m_config) in AutoModelForImageTextToText._model_mapping.keys():
            model_class = AutoModelForImageTextToText
        elif type(m_config) in AutoModelForMaskedLM._model_mapping.keys():
            model_class = AutoModelForMaskedLM
        elif type(m_config) in AutoModel._model_mapping.keys():
            model_class = AutoModel
    else:
        raise ValueError(f"Model type '{model_type}' is not found in CONFIG_MAPPING.")
    return model_class, m_config
