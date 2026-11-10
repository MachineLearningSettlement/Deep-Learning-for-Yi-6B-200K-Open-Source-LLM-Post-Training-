"""Reusable model-loading and LoRA utilities for Yi-6B-200K post-training."""

import os
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve an explicit device or select CUDA when available."""
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_precision(
    precision: str,
    device: Union[str, torch.device],
) -> Tuple[torch.dtype, bool]:
    """Return the model dtype and whether FP16 gradient scaling is required."""
    device = resolve_device(device)

    if precision == "auto":
        if device.type == "cuda":
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            precision = "fp32"

    if device.type != "cuda" and precision in ("fp16", "bf16"):
        raise ValueError("FP16 and BF16 training require a CUDA device.")

    if precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The current CUDA device does not support BF16.")
        return torch.bfloat16, False

    if precision == "fp16":
        return torch.float16, True

    if precision == "fp32":
        return torch.float32, False

    raise ValueError(f"Unsupported precision: {precision}")


def load_model_and_tokenizer(
    model_name_or_path: str = "01-ai/Yi-6B-200K",
    device: Optional[Union[str, torch.device]] = None,
    precision: str = "auto",
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = True,
):
    """Load a causal language model and tokenizer with the project's SFT settings.

    Returns:
        model, tokenizer, model_dtype, use_fp16_scaler
    """
    device = resolve_device(device)
    model_dtype, use_fp16_scaler = resolve_precision(precision, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {"torch_dtype": model_dtype}
    if device.type == "cuda":
        model_kwargs["device_map"] = "auto"

    if load_in_4bit:
        if device.type != "cuda":
            raise RuntimeError("4-bit loading requires a CUDA device.")

        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model, tokenizer, model_dtype, use_fp16_scaler


def apply_lora(
    model,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    """Apply LoRA to the selected causal-language-model projection modules."""
    if rank < 1 or alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1).")

    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    available_modules = {
        name.rsplit(".", 1)[-1] for name, _ in model.named_modules()
    }
    missing_modules = set(target_modules) - available_modules
    if missing_modules:
        raise ValueError(
            f"LoRA target modules not found: {sorted(missing_modules)}"
        )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    adapted_model = get_peft_model(model, lora_config)
    adapted_model.print_trainable_parameters()
    return adapted_model


def save_adapter_and_tokenizer(
    model,
    tokenizer,
    output_dir: str,
    safe_serialization: bool = True,
) -> str:
    """Save a trained PEFT adapter and its tokenizer to output_dir."""
    if model is None or tokenizer is None:
        raise ValueError("Both model and tokenizer are required.")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(output_dir)
    return output_dir
