"""Model configurations used by MO-CAPO and Tri-Fair experiments."""

from __future__ import annotations
from pathlib import Path
from typing import Dict
from tri_fair.config.schema import ModelConfig, ModelType

_GPT_OSS_120B = ModelConfig(
    model="vllm-openai/gpt-oss-120b",
    alias="gpt-oss-120b",
    max_model_len=3000,
    batch_size=14,
    input_costs=0.12,
    output_costs=0.49,
    model_storage_path=Path("../models/gpt-oss-120b/"),
    llm_kwargs={"revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"},
)
_QWEN_3_30B = ModelConfig(
    model="vllm-Qwen/Qwen3-30B-A3B-Instruct-2507",
    alias="qwen-3-30b",
    max_model_len=3000,
    batch_size=48,
    input_costs=0.11,
    output_costs=0.41,
    model_storage_path=Path("../models/Qwen3-30B/"),
    llm_kwargs={"revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"},
)
_MISTRAL_3_24B = ModelConfig(
    model="vllm-mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    alias="mistral-3-24b",
    max_model_len=3000,
    batch_size=16,
    input_costs=0.08,
    output_costs=0.32,
    model_storage_path=Path("../models/Mistral-3-24b/"),
    llm_kwargs={
        "revision": "95a6d26c4bfb886c58daf9d3f7332c857cb27b43",
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "dtype": "bfloat16",
    },
)
ALL_MODELS: Dict[ModelType, ModelConfig] = {
    config.alias: config for config in (_GPT_OSS_120B, _QWEN_3_30B, _MISTRAL_3_24B)
}
for _model_config in ALL_MODELS.values():
    _model_config.validate()
