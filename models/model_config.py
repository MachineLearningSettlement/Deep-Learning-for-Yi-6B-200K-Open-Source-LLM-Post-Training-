"""Configuration for the pretrained Yi-6B-200K model."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "01-ai/Yi-6B-200K"
    max_sequence_length: int = 2048
    use_fast_tokenizer: bool = True
    tokenizer_padding_side: str = "right"
    use_cache: bool = False
    default_precision: str = "auto"
    load_in_4bit: bool = False
    quantization_type: str = "nf4"
    use_double_quantization: bool = True

    @property
    def model_name_or_path(self) -> str:
        """Return a local model path when YI_MODEL_PATH is set; otherwise use the HF ID."""
        return os.environ.get("YI_MODEL_PATH", self.model_id)


MODEL_CONFIG = ModelConfig()

MODEL_ID = MODEL_CONFIG.model_id
MODEL_NAME_OR_PATH = MODEL_CONFIG.model_name_or_path
MAX_SEQUENCE_LENGTH = MODEL_CONFIG.max_sequence_length
USE_FAST_TOKENIZER = MODEL_CONFIG.use_fast_tokenizer
TOKENIZER_PADDING_SIDE = MODEL_CONFIG.tokenizer_padding_side
USE_CACHE = MODEL_CONFIG.use_cache
DEFAULT_PRECISION = MODEL_CONFIG.default_precision
LOAD_IN_4BIT = MODEL_CONFIG.load_in_4bit
QUANTIZATION_TYPE = MODEL_CONFIG.quantization_type
USE_DOUBLE_QUANTIZATION = MODEL_CONFIG.use_double_quantization


def get_model_name_or_path(override: Optional[str] = None) -> str:
    """Resolve an explicit path, environment override, or Hugging Face model ID."""
    if override:
        return override
    return MODEL_CONFIG.model_name_or_path
