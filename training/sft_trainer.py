
"""
Supervised Fine-Tuning (SFT) of Yi-6B-200K using LoRA
======================================================

Project:
    Deep Learning Framework for Post-Training the Yi-6B-200K
    Open-Source LLM

Purpose:
    1. Load the pretrained Yi-6B-200K causal language model.
    2. Load curated prompt-response training examples.
    3. Tokenize instruction and response sequences.
    4. Mask prompt tokens during loss computation.
    5. Configure Low-Rank Adaptation (LoRA).
    6. Fine-tune the model using Supervised Fine-Tuning.
    7. Apply gradient accumulation and mixed precision.
    8. Evaluate validation loss.
    9. Apply early stopping.
    10. Save the trained LoRA adapter and tokenizer.

Training objective:
    Given a prompt x and target response y, optimize:

        L_SFT = - SUM log P(y_t | x, y_<t)

    The prompt tokens are masked from the loss.

Model:
    01-ai/Yi-6B-200K

Fine-tuning:
    LoRA / PEFT

Input:
    JSONL dataset containing:

        {
            "sample_id": "S001",
            "prompt": "...",
            "response": "..."
        }

    An optional NumPy file containing curated sample indices.

Output:
    - LoRA adapter weights.
    - Adapter configuration.
    - Tokenizer files.
    - Training history.
    - Training configuration.

Requirements:
    pip install torch transformers peft accelerate numpy

For 4-bit quantization:
    pip install bitsandbytes

Author:
    PlotDot-LLM Project
"""

import os
import json
import copy
import random
import argparse

from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Subset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. RAW DATASET LOADING
# ============================================================

def load_jsonl(
    path: str,
    curated_indices: Optional[str] = None
) -> List[Dict]:

    """
    Load prompt-response examples from JSONL.

    Optional curated_indices:
        NumPy file containing the original sample indices
        selected by dataset_curator.py.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    records = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:

                record = json.loads(line)

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                )

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            if "prompt" not in record or "response" not in record:
                raise ValueError(
                    f"Line {line_number} must contain prompt and response."
                )

            prompt = record["prompt"]
            response = record["response"]

            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Invalid prompt at line {line_number}."
                )

            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    f"Invalid response at line {line_number}."
                )

            records.append(record)

    if not records:
        raise ValueError(
            "The input dataset contains no training examples."
        )

    if curated_indices is not None:

        indices = np.load(
            curated_indices,
            allow_pickle=False
        )

        indices = np.asarray(indices)

        if indices.ndim != 1:
            raise ValueError(
                "Curated indices must be a one-dimensional array."
            )

        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(
                "Curated indices must contain integers."
            )

        if len(indices) == 0:
            raise ValueError(
                "The curated index array is empty."
            )

        if len(np.unique(indices)) != len(indices):
            raise ValueError(
                "Curated indices must be unique."
            )

        if np.any(indices < 0) or np.any(indices >= len(records)):
            raise IndexError(
                "Curated indices exceed the original dataset boundaries."
            )

        records = [
            records[int(index)]
            for index in indices
        ]

    if len(records) < 2:
        raise ValueError(
            "At least two prompt-response examples are required."
        )

    return records


# ============================================================
# 3. TOKENIZATION AND SFT DATASET
# ============================================================

class SFTDataset(Dataset):

    """
    Tokenized prompt-response dataset.

    Prompt tokens:
        labels = -100

    Response tokens:
        labels = input_ids

    This ensures that the language-modeling loss is calculated
    only on response tokens.
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        max_seq_length: int = 2048,
        prompt_template: str = (
            "### Instruction:\n"
            "{prompt}\n\n"
            "### Response:\n"
        )
    ):

        super().__init__()

        if max_seq_length < 2:
            raise ValueError(
                "max_seq_length must be at least 2."
            )

        if "{prompt}" not in prompt_template:
            raise ValueError(
                "prompt_template must contain {prompt}."
            )

        if tokenizer.eos_token_id is None:
            raise ValueError(
                "The tokenizer must define an EOS token."
            )

        self.tokenizer = tokenizer

        self.max_seq_length = max_seq_length

        self.prompt_template = prompt_template

        self.examples = []

        for record in records:

            prompt = record["prompt"]

            response = record["response"]

            # ------------------------------------------------
            # Tokenize prompt
            # ------------------------------------------------

            prompt_text = prompt_template.format(
                prompt=prompt
            )

            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=False
            )["input_ids"]

            # ------------------------------------------------
            # Tokenize response
            # ------------------------------------------------

            response_ids = tokenizer(
                response,
                add_special_tokens=False,
                truncation=False
            )["input_ids"]

            if not response_ids:
                raise ValueError(
                    "A response produced an empty token sequence."
                )

            eos_token_id = tokenizer.eos_token_id

            # ------------------------------------------------
            # Preserve response tokens and EOS
            # ------------------------------------------------

            response_budget = max_seq_length - 1

            if len(response_ids) > response_budget:

                response_ids = response_ids[
                    :response_budget
                ]

            response_ids.append(
                eos_token_id
            )

            # ------------------------------------------------
            # Truncate prompt while preserving the response
            # ------------------------------------------------

            prompt_budget = (
                max_seq_length - len(response_ids)
            )

            if prompt_budget < 0:
                raise RuntimeError(
                    "Invalid prompt-response token budget."
                )

            if len(prompt_ids) > prompt_budget:

                if (
                    tokenizer.bos_token_id is not None
                    and prompt_budget > 1
                ):

                    prompt_ids = (
                        [tokenizer.bos_token_id]
                        + prompt_ids[-(prompt_budget - 1):]
                    )

                else:

                    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []

            # ------------------------------------------------
            # Construct causal language modeling sequence
            # ------------------------------------------------

            input_ids = (
                prompt_ids + response_ids
            )

            labels = (
                [-100] * len(prompt_ids)
                + response_ids
            )

            attention_mask = [
                1
            ] * len(input_ids)

            if len(input_ids) > max_seq_length:
                raise RuntimeError(
                    "Tokenized sequence exceeds max_seq_length."
                )

            self.examples.append({

                "input_ids": input_ids,

                "attention_mask": attention_mask,

                "labels": labels

            })

    def __len__(self) -> int:

        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:

        example = self.examples[index]

        return {

            "input_ids": torch.tensor(
                example["input_ids"],
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                example["attention_mask"],
                dtype=torch.long
            ),

            "labels": torch.tensor(
                example["labels"],
                dtype=torch.long
            )

        }


# ============================================================
# 4. SFT DATA COLLATOR
# ============================================================

class SFTDataCollator:

    """
    Dynamically pad tokenized sequences within each batch.

    Input IDs:
        Padded using the tokenizer's padding token.

    Attention masks:
        Padded using zero.

    Labels:
        Padded using -100 so padding positions do not
        contribute to the training objective.
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_to_multiple_of: Optional[int] = 8
    ):

        self.pad_token_id = pad_token_id

        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(
        self,
        batch: List[Dict]
    ) -> Dict[str, torch.Tensor]:

        max_length = max(
            len(item["input_ids"])
            for item in batch
        )

        if self.pad_to_multiple_of is not None:

            multiple = self.pad_to_multiple_of

            max_length = (
                (max_length + multiple - 1)
                // multiple
            ) * multiple

        batch_size = len(batch)

        input_ids = torch.full(
            (batch_size, max_length),
            fill_value=self.pad_token_id,
            dtype=torch.long
        )

        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long
        )

        labels = torch.full(
            (batch_size, max_length),
            fill_value=-100,
            dtype=torch.long
        )

        for index, item in enumerate(batch):

            sequence_length = len(
                item["input_ids"]
            )

            input_ids[index, :sequence_length] = (
                item["input_ids"]
            )

            attention_mask[index, :sequence_length] = (
                item["attention_mask"]
            )

            labels[index, :sequence_length] = (
                item["labels"]
            )

        return {

            "input_ids": input_ids,

            "attention_mask": attention_mask,

            "labels": labels

        }


# ============================================================
# 5. MODEL AND TOKENIZER LOADING
# ============================================================

def resolve_precision(
    precision: str,
    device: torch.device
) -> Tuple[torch.dtype, bool]:

    """
    Resolve numerical precision.

    Returns:
        Model dtype.
        Whether FP16 gradient scaling is enabled.
    """

    if precision == "auto":

        if device.type == "cuda":

            if torch.cuda.is_bf16_supported():
                precision = "bf16"

            else:
                precision = "fp16"

        else:

            precision = "fp32"

    if device.type != "cuda" and precision in ("fp16", "bf16"):

        raise ValueError(
            "FP16 and BF16 training require CUDA."
        )

    if precision == "bf16":

        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "The current CUDA device does not support BF16."
            )

        return torch.bfloat16, False

    if precision == "fp16":

        return torch.float16, True

    if precision == "fp32":

        return torch.float32, False

    raise ValueError(
        f"Unsupported precision: {precision}"
    )


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: torch.device,
    precision: str,
    load_in_4bit: bool,
    gradient_checkpointing: bool
):

    """
    Load Yi-6B-200K and configure its tokenizer.

    Supports:
        - Full-precision loading.
        - FP16 loading.
        - BF16 loading.
        - Optional 4-bit quantization.
        - Gradient checkpointing.
    """

    model_dtype, use_fp16_scaler = resolve_precision(
        precision,
        device
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model_kwargs = {

        "torch_dtype": model_dtype

    }

    if device.type == "cuda":

        model_kwargs["device_map"] = "auto"

    if load_in_4bit:

        if device.type != "cuda":
            raise RuntimeError(
                "4-bit loading requires a CUDA device."
            )

        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(

            load_in_4bit=True,

            bnb_4bit_quant_type="nf4",

            bnb_4bit_compute_dtype=model_dtype,

            bnb_4bit_use_double_quant=True

        )

        model_kwargs["quantization_config"] = (
            quantization_config
        )

    print(
        f"Loading model: {model_name_or_path}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )

    if model.config.pad_token_id is None:

        model.config.pad_token_id = (
            tokenizer.pad_token_id
        )

    model.config.use_cache = False

    if load_in_4bit:

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing
        )

    elif gradient_checkpointing:

        model.gradient_checkpointing_enable()

        if hasattr(
            model,
            "enable_input_require_grads"
        ):

            model.enable_input_require_grads()

    return (
        model,
        tokenizer,
        model_dtype,
        use_fp16_scaler
    )


# ============================================================
# 6. LORA CONFIGURATION
# ============================================================

def apply_lora(
    model,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None
):

    """
    Apply Low-Rank Adaptation to the pretrained model.

    LoRA introduces trainable low-rank matrices into selected
    Transformer linear projection modules.
    """

    if rank < 1 or alpha < 1:
        raise ValueError(
            "LoRA rank and alpha must be positive."
        )

    if not 0 <= dropout < 1:
        raise ValueError(
            "LoRA dropout must be in [0, 1)."
        )

    if target_modules is None:

        target_modules = [

            "q_proj",

            "k_proj",

            "v_proj",

            "o_proj",

            "gate_proj",

            "up_proj",

            "down_proj"

        ]

    available_modules = {
        name.split(".")[-1]
        for name, _ in model.named_modules()
    }

    missing_modules = (
        set(target_modules) - available_modules
    )

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

        inference_mode=False

    )

    model = get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    return model


# ============================================================
# 7. SFT TRAINER
# ============================================================

class SFTTrainer:

    """
    Complete Supervised Fine-Tuning pipeline.

    Responsibilities:

        - Model loading.
        - Tokenization.
        - LoRA configuration.
        - Training and validation splitting.
        - Response-only loss calculation.
        - Gradient accumulation.
        - Mixed precision.
        - Learning-rate scheduling.
        - Gradient clipping.
        - Early stopping.
        - Adapter checkpoint saving.
    """

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str,
        max_seq_length: int = 2048,
        learning_rate: float = 2e-4,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        epochs: int = 3,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        validation_split: float = 0.1,
        patience: int = 3,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        precision: str = "auto",
        load_in_4bit: bool = False,
        gradient_checkpointing: bool = True,
        device: Optional[str] = None,
        seed: int = 42
    ):

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        if gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be positive."
            )

        if epochs < 1 or patience < 1:
            raise ValueError(
                "epochs and patience must be positive."
            )

        if not 0 <= warmup_ratio < 1:
            raise ValueError(
                "warmup_ratio must be in [0, 1)."
            )

        if weight_decay < 0:
            raise ValueError(
                "weight_decay must be non-negative."
            )

        if max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be positive."
            )

        if not 0 < validation_split < 1:
            raise ValueError(
                "validation_split must be between 0 and 1."
            )

        set_seed(seed)

        self.model_name_or_path = model_name_or_path

        self.output_dir = output_dir

        self.max_seq_length = max_seq_length

        self.learning_rate = learning_rate

        self.batch_size = batch_size

        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )

        self.epochs = epochs

        self.warmup_ratio = warmup_ratio

        self.weight_decay = weight_decay

        self.max_grad_norm = max_grad_norm

        self.validation_split = validation_split

        self.patience = patience

        self.lora_rank = lora_rank

        self.lora_alpha = lora_alpha

        self.lora_dropout = lora_dropout

        self.lora_target_modules = lora_target_modules

        self.precision = precision

        self.load_in_4bit = load_in_4bit

        self.gradient_checkpointing = gradient_checkpointing

        self.seed = seed

        self.device = torch.device(
            device if device else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Load pretrained model and tokenizer
        # ----------------------------------------------------

        (
            base_model,
            self.tokenizer,
            self.model_dtype,
            self.use_fp16_scaler
        ) = load_model_and_tokenizer(

            model_name_or_path=model_name_or_path,

            device=self.device,

            precision=precision,

            load_in_4bit=load_in_4bit,

            gradient_checkpointing=gradient_checkpointing

        )

        # ----------------------------------------------------
        # Apply LoRA
        # ----------------------------------------------------

        self.model = apply_lora(

            model=base_model,

            rank=lora_rank,

            alpha=lora_alpha,

            dropout=lora_dropout,

            target_modules=lora_target_modules

        )

        self.model.config.use_cache = False

        self.model_device = (
            self.model.get_input_embeddings().weight.device
        )

        if self.model_device.type == "meta":
            raise RuntimeError(
                "The model input embeddings are not initialized."
            )

        self.history = {

            "train_loss": [],

            "validation_loss": [],

            "learning_rate": []

        }

        self.is_fitted = False

        self.best_validation_loss = float("inf")

    # --------------------------------------------------------
    # 7.1 DATASET PREPARATION
    # --------------------------------------------------------

    def prepare_dataset(
        self,
        records: List[Dict],
        prompt_template: str
    ) -> Tuple[DataLoader, DataLoader]:

        dataset = SFTDataset(

            records=records,

            tokenizer=self.tokenizer,

            max_seq_length=self.max_seq_length,

            prompt_template=prompt_template

        )

        if len(dataset) < 2:
            raise ValueError(
                "At least two examples are required for training."
            )

        rng = np.random.default_rng(
            self.seed
        )

        indices = rng.permutation(
            len(dataset)
        )

        n_validation = max(
            1,
            int(len(dataset) * self.validation_split)
        )

        n_validation = min(
            n_validation,
            len(dataset) - 1
        )

        validation_indices = indices[
            :n_validation
        ].tolist()

        training_indices = indices[
            n_validation:
        ].tolist()

        training_subset = Subset(
            dataset,
            training_indices
        )

        validation_subset = Subset(
            dataset,
            validation_indices
        )

        collator = SFTDataCollator(
            pad_token_id=self.tokenizer.pad_token_id
        )

        training_loader = DataLoader(

            training_subset,

            batch_size=self.batch_size,

            shuffle=True,

            collate_fn=collator,

            pin_memory=self.device.type == "cuda"

        )

        validation_loader = DataLoader(

            validation_subset,

            batch_size=self.batch_size,

            shuffle=False,

            collate_fn=collator,

            pin_memory=self.device.type == "cuda"

        )

        return (
            training_loader,
            validation_loader
        )

    # --------------------------------------------------------
    # 7.2 MIXED PRECISION CONTEXT
    # --------------------------------------------------------

    def autocast_context(self):

        if self.device.type != "cuda":

            return nullcontext()

        if self.model_dtype in (
            torch.float16,
            torch.bfloat16
        ):

            return torch.autocast(
                device_type="cuda",
                dtype=self.model_dtype
            )

        return nullcontext()

    # --------------------------------------------------------
    # 7.3 VALIDATION
    # --------------------------------------------------------

    def evaluate(
        self,
        validation_loader: DataLoader
    ) -> float:

        self.model.eval()

        total_loss = 0.0

        total_tokens = 0

        with torch.inference_mode():

            for batch in validation_loader:

                input_ids = batch["input_ids"].to(
                    self.model_device
                )

                attention_mask = batch["attention_mask"].to(
                    self.model_device
                )

                labels = batch["labels"].to(
                    self.model_device
                )

                supervised_tokens = (
                    labels[:, 1:] != -100
                ).sum().item()

                if supervised_tokens == 0:
                    continue

                with self.autocast_context():

                    outputs = self.model(

                        input_ids=input_ids,

                        attention_mask=attention_mask,

                        labels=labels

                    )

                    loss = outputs.loss

                total_loss += (
                    loss.item() * supervised_tokens
                )

                total_tokens += supervised_tokens

        if total_tokens == 0:
            raise RuntimeError(
                "Validation dataset contains no supervised response tokens."
            )

        return total_loss / total_tokens

    # --------------------------------------------------------
    # 7.4 TRAINING
    # --------------------------------------------------------

    def fit(
        self,
        records: List[Dict],
        prompt_template: str = (
            "### Instruction:\n"
            "{prompt}\n\n"
            "### Response:\n"
        )
    ) -> Dict[str, List[float]]:

        training_loader, validation_loader = (
            self.prepare_dataset(
                records,
                prompt_template
            )
        )

        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise RuntimeError(
                "No trainable parameters were found."
            )

        optimizer = torch.optim.AdamW(

            trainable_parameters,

            lr=self.learning_rate,

            weight_decay=self.weight_decay

        )

        updates_per_epoch = (
            len(training_loader)
            + self.gradient_accumulation_steps
            - 1
        ) // self.gradient_accumulation_steps

        total_training_steps = (
            updates_per_epoch * self.epochs
        )

        warmup_steps = int(
            total_training_steps * self.warmup_ratio
        )

        scheduler = get_linear_schedule_with_warmup(

            optimizer,

            num_warmup_steps=warmup_steps,

            num_training_steps=total_training_steps

        )

        scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_fp16_scaler
        )

        best_adapter_state = None

        epochs_without_improvement = 0

        global_step = 0

        print("\nStarting Supervised Fine-Tuning")

        print(
            f"Training examples: {len(training_loader.dataset)}"
        )

        print(
            f"Validation examples: {len(validation_loader.dataset)}"
        )

        print(
            f"Gradient accumulation: "
            f"{self.gradient_accumulation_steps}"
        )

        print(
            f"Total optimizer steps: {total_training_steps}"
        )

        print(
            f"Precision: {self.model_dtype}"
        )

        print("-" * 85)

        # ====================================================
        # EPOCH LOOP
        # ====================================================

        for epoch in range(self.epochs):

            self.model.train()

            optimizer.zero_grad(
                set_to_none=True
            )

            epoch_loss = 0.0

            epoch_tokens = 0

            accumulation_count = 0

            number_batches = len(
                training_loader
            )

            # ================================================
            # TRAINING PHASE
            # ================================================

            for batch_index, batch in enumerate(
                training_loader
            ):

                input_ids = batch["input_ids"].to(
                    self.model_device
                )

                attention_mask = batch["attention_mask"].to(
                    self.model_device
                )

                labels = batch["labels"].to(
                    self.model_device
                )

                supervised_tokens = (
                    labels[:, 1:] != -100
                ).sum().item()

                if supervised_tokens == 0:
                    continue

                with self.autocast_context():

                    outputs = self.model(

                        input_ids=input_ids,

                        attention_mask=attention_mask,

                        labels=labels

                    )

                    loss = outputs.loss

                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)

                    raise RuntimeError(
                        "Non-finite loss detected during SFT."
                    )

                scaler.scale(
                    loss / self.gradient_accumulation_steps
                ).backward()

                accumulation_count += 1

                epoch_loss += (
                    loss.item() * supervised_tokens
                )

                epoch_tokens += supervised_tokens

                is_accumulation_step = (
                    accumulation_count
                    == self.gradient_accumulation_steps
                )

                is_last_batch = (
                    batch_index == number_batches - 1
                )

                if is_accumulation_step or is_last_batch:

                    if accumulation_count < self.gradient_accumulation_steps:

                        correction = (
                            self.gradient_accumulation_steps
                            / accumulation_count
                        )

                        for parameter in trainable_parameters:

                            if parameter.grad is not None:

                                parameter.grad.mul_(
                                    correction
                                )

                    scaler.unscale_(
                        optimizer
                    )

                    nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=self.max_grad_norm
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                    scheduler.step()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    accumulation_count = 0

                    global_step += 1

            if epoch_tokens == 0:
                raise RuntimeError(
                    "Training dataset contains no supervised response tokens."
                )

            train_loss = (
                epoch_loss / epoch_tokens
            )

            # ================================================
            # VALIDATION PHASE
            # ================================================

            validation_loss = self.evaluate(
                validation_loader
            )

            current_lr = scheduler.get_last_lr()[0]

            self.history["train_loss"].append(
                float(train_loss)
            )

            self.history["validation_loss"].append(
                float(validation_loss)
            )

            self.history["learning_rate"].append(
                float(current_lr)
            )

            print(

                f"Epoch [{epoch + 1:03d}/{self.epochs}] "

                f"| Train Loss: {train_loss:.6f} "

                f"| Validation Loss: {validation_loss:.6f} "

                f"| Learning Rate: {current_lr:.8f}"

            )

            # ================================================
            # EARLY STOPPING
            # ================================================

            if validation_loss < self.best_validation_loss:

                self.best_validation_loss = validation_loss

                best_adapter_state = {

                    name: parameter.detach().cpu().clone()

                    for name, parameter in self.model.named_parameters()

                    if parameter.requires_grad

                }

                epochs_without_improvement = 0

            else:

                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:

                print(
                    "Early stopping triggered."
                )

                break

        if best_adapter_state is None:

            raise RuntimeError(
                "Training did not produce a valid adapter checkpoint."
            )

        # ----------------------------------------------------
        # Restore best LoRA parameters
        # ----------------------------------------------------

        for name, parameter in self.model.named_parameters():

            if parameter.requires_grad:

                parameter.data.copy_(
                    best_adapter_state[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype
                    )
                )

        self.model.eval()

        self.model.config.use_cache = True

        self.is_fitted = True

        print("-" * 85)

        print(
            f"Best validation loss: "
            f"{self.best_validation_loss:.6f}"
        )

        print(
            f"Total optimizer steps: {global_step}"
        )

        print(
            "Supervised Fine-Tuning completed."
        )

        return self.history

    # --------------------------------------------------------
    # 7.5 SAVE LORA ADAPTER
    # --------------------------------------------------------

    def save(self) -> None:

        if not self.is_fitted:
            raise RuntimeError(
                "The model must be trained before saving."
            )

        adapter_dir = os.path.join(
            self.output_dir,
            "adapter"
        )

        os.makedirs(
            adapter_dir,
            exist_ok=True
        )

        self.model.save_pretrained(
            adapter_dir,
            safe_serialization=True
        )

        self.tokenizer.save_pretrained(
            adapter_dir
        )

        # ----------------------------------------------------
        # Training history
        # ----------------------------------------------------

        history_path = os.path.join(
            self.output_dir,
            "training_history.json"
        )

        with open(
            history_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.history,
                file,
                indent=4
            )

        # ----------------------------------------------------
        # Training configuration
        # ----------------------------------------------------

        training_config = {

            "base_model": self.model_name_or_path,

            "fine_tuning_method": "LoRA",

            "task_type": "CAUSAL_LM",

            "lora_rank": self.lora_rank,

            "lora_alpha": self.lora_alpha,

            "lora_dropout": self.lora_dropout,

            "learning_rate": self.learning_rate,

            "batch_size": self.batch_size,

            "gradient_accumulation_steps":
                self.gradient_accumulation_steps,

            "epochs": self.epochs,

            "max_seq_length": self.max_seq_length,

            "validation_split": self.validation_split,

            "precision": str(self.model_dtype),

            "load_in_4bit": self.load_in_4bit,

            "best_validation_loss":
                self.best_validation_loss

        }

        config_path = os.path.join(
            self.output_dir,
            "training_config.json"
        )

        with open(
            config_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                training_config,
                file,
                indent=4
            )

        print("\nSFT adapter saved successfully.")

        print(
            f"Adapter directory: {adapter_dir}"
        )


# ============================================================
# 8. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Supervised Fine-Tuning of Yi-6B-200K "
            "using LoRA and PEFT."
        )
    )

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="./models/Yi-6B-200K"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )

    parser.add_argument(
        "--curated-indices",
        type=str,
        default=None
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/sft"
    )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048
    )

    parser.add_argument(
        "--prompt-template",
        type=str,
        default=(
            "### Instruction:\n"
            "{prompt}\n\n"
            "### Response:\n"
        )
    )

    # --------------------------------------------------------
    # LoRA configuration
    # --------------------------------------------------------

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3
    )

    # --------------------------------------------------------
    # Memory and precision
    # --------------------------------------------------------

    parser.add_argument(
        "--precision",
        type=str,
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto"
    )

    parser.add_argument(
        "--load-in-4bit",
        action="store_true"
    )

    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true"
    )

    # --------------------------------------------------------
    # Runtime configuration
    # --------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Load training records
    # --------------------------------------------------------

    records = load_jsonl(

        path=args.dataset,

        curated_indices=args.curated_indices

    )

    print(
        f"Loaded {len(records)} SFT examples."
    )

    # --------------------------------------------------------
    # Parse LoRA target modules
    # --------------------------------------------------------

    target_modules = [

        module.strip()

        for module in args.lora_target_modules.split(",")

        if module.strip()

    ]

    # --------------------------------------------------------
    # Initialize trainer
    # --------------------------------------------------------

    trainer = SFTTrainer(

        model_name_or_path=args.model_name_or_path,

        output_dir=args.output_dir,

        max_seq_length=args.max_seq_length,

        learning_rate=args.learning_rate,

        batch_size=args.batch_size,

        gradient_accumulation_steps=args.gradient_accumulation_steps,

        epochs=args.epochs,

        warmup_ratio=args.warmup_ratio,

        weight_decay=args.weight_decay,

        max_grad_norm=args.max_grad_norm,

        validation_split=args.validation_split,

        patience=args.patience,

        lora_rank=args.lora_rank,

        lora_alpha=args.lora_alpha,

        lora_dropout=args.lora_dropout,

        lora_target_modules=target_modules,

        precision=args.precision,

        load_in_4bit=args.load_in_4bit,

        gradient_checkpointing=(
            not args.no_gradient_checkpointing
        ),

        device=args.device,

        seed=args.seed

    )

    # --------------------------------------------------------
    # Run SFT
    # --------------------------------------------------------

    trainer.fit(

        records=records,

        prompt_template=args.prompt_template

    )

    # --------------------------------------------------------
    # Save trained adapter
    # --------------------------------------------------------

    trainer.save()


# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
