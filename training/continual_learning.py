"""Continual SFT for Yi-6B-200K using adaptive skill replay.

The trainer mixes narrative examples with retention examples representing
capabilities acquired by the pretrained model. It adjusts the retention replay
ratio according to capability deviation and narrative plasticity, and monitors
stability across adaptation rounds.

Input JSONL files must contain prompt and response fields. Four separate files
are used so that training and evaluation examples remain independent:

    narrative_train.jsonl
    retention_train.jsonl
    narrative_eval.jsonl
    retention_eval.jsonl
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

try:
    from .sft_trainer import (
        SFTDataCollator,
        SFTDataset,
        apply_lora,
        load_jsonl,
        load_model_and_tokenizer,
        set_seed,
    )
except ImportError:
    from sft_trainer import (
        SFTDataCollator,
        SFTDataset,
        apply_lora,
        load_jsonl,
        load_model_and_tokenizer,
        set_seed,
    )


DEFAULT_PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "{prompt}\n\n"
    "### Response:\n"
)


def capability_deviation(
    baseline_retention_loss: float,
    current_retention_loss: float,
) -> float:
    """Return relative deterioration on retained capabilities (lower is better).

    Improvements over the initial model produce zero deviation rather than a
    negative forgetting score.
    """
    if baseline_retention_loss <= 0:
        raise ValueError("Baseline retention loss must be positive.")

    deterioration = (
        current_retention_loss - baseline_retention_loss
    ) / baseline_retention_loss

    return float(max(0.0, deterioration))


def plasticity_score(
    baseline_narrative_loss: float,
    current_narrative_loss: float,
) -> float:
    """Return relative narrative-loss reduction (higher is better).

    Negative values indicate that narrative evaluation loss increased relative
    to the pretrained model's initial loss.
    """
    if baseline_narrative_loss <= 0:
        raise ValueError("Baseline narrative loss must be positive.")

    return float(
        (baseline_narrative_loss - current_narrative_loss)
        / baseline_narrative_loss
    )


def stability_score(
    previous_capability_deviation: Optional[float],
    current_capability_deviation: float,
    stability_scale: float,
) -> float:
    """Measure round-to-round stability of capability deviation in [0, 1].

    A score of 1 means capability deviation did not change from the previous
    round. A score of 0 means the change was at least stability_scale.
    """
    if stability_scale <= 0:
        raise ValueError("stability_scale must be positive.")

    if previous_capability_deviation is None:
        return 0.0

    change = abs(
        current_capability_deviation - previous_capability_deviation
    )

    return float(max(0.0, 1.0 - change / stability_scale))


def atomic_json_save(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    os.replace(temporary_path, path)


class ContinualLearningTrainer:
    """Adapt a single LoRA model using adaptive narrative/retention replay."""

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str,
        max_seq_length: int = 2048,
        learning_rate: float = 2e-4,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        epochs_per_round: int = 1,
        max_rounds: int = 10,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        initial_replay_ratio: float = 0.25,
        min_replay_ratio: float = 0.05,
        max_replay_ratio: float = 0.80,
        replay_ratio_step: float = 0.05,
        max_capability_deviation: float = 0.05,
        min_plasticity: float = 0.05,
        min_stability: float = 0.90,
        stability_scale: float = 0.05,
        convergence_patience: int = 2,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        precision: str = "auto",
        load_in_4bit: bool = False,
        gradient_checkpointing: bool = True,
        device: Optional[str] = None,
        seed: int = 42,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if batch_size < 1 or gradient_accumulation_steps < 1:
            raise ValueError("Batch size and gradient accumulation must be positive.")
        if epochs_per_round < 1 or max_rounds < 1:
            raise ValueError("epochs_per_round and max_rounds must be positive.")
        if not 0 <= warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1).")
        if weight_decay < 0 or max_grad_norm <= 0:
            raise ValueError("weight_decay must be non-negative and max_grad_norm positive.")
        if not 0 <= min_replay_ratio <= initial_replay_ratio <= max_replay_ratio < 1:
            raise ValueError(
                "Replay ratios must satisfy 0 <= min <= initial <= max < 1."
            )
        if replay_ratio_step <= 0:
            raise ValueError("replay_ratio_step must be positive.")
        if max_capability_deviation < 0 or min_stability < 0 or min_stability > 1:
            raise ValueError("Invalid capability-deviation or stability threshold.")
        if stability_scale <= 0:
            raise ValueError("stability_scale must be positive.")
        if convergence_patience < 1:
            raise ValueError("convergence_patience must be positive.")
        if not 0 <= min_plasticity:
            raise ValueError("min_plasticity must be non-negative.")
        if not 0 <= lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1).")

        set_seed(seed)
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs_per_round = epochs_per_round
        self.max_rounds = max_rounds
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.initial_replay_ratio = initial_replay_ratio
        self.min_replay_ratio = min_replay_ratio
        self.max_replay_ratio = max_replay_ratio
        self.replay_ratio_step = replay_ratio_step
        self.max_capability_deviation = max_capability_deviation
        self.min_plasticity = min_plasticity
        self.min_stability = min_stability
        self.stability_scale = stability_scale
        self.convergence_patience = convergence_patience
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.precision = precision
        self.load_in_4bit = load_in_4bit
        self.gradient_checkpointing = gradient_checkpointing
        self.seed = seed
        self.prompt_template = prompt_template
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        os.makedirs(output_dir, exist_ok=True)

        (
            base_model,
            self.tokenizer,
            self.model_dtype,
            self.use_fp16_scaler,
        ) = load_model_and_tokenizer(
            model_name_or_path=model_name_or_path,
            device=self.device,
            precision=precision,
            load_in_4bit=load_in_4bit,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.model = apply_lora(
            model=base_model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        self.model.config.use_cache = False
        self.model_device = self.model.get_input_embeddings().weight.device

        if self.model_device.type == "meta":
            raise RuntimeError("The model input embeddings are not initialized.")

        self.collator = SFTDataCollator(
            pad_token_id=self.tokenizer.pad_token_id
        )
        self.history = {
            "baseline": {},
            "rounds": [],
            "converged": False,
            "final_replay_ratio": float(initial_replay_ratio),
        }

    def _make_dataset(self, records: List[Dict]) -> SFTDataset:
        return SFTDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            prompt_template=self.prompt_template,
        )

    def _make_eval_loader(self, dataset: SFTDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            pin_memory=self.model_device.type == "cuda",
        )

    def _make_mixed_training_loader(
        self,
        narrative_dataset: SFTDataset,
        retention_dataset: SFTDataset,
        replay_ratio: float,
        round_index: int,
        epoch_index: int,
    ) -> Tuple[DataLoader, int, int]:
        """Use every narrative example once and sample retention examples to ratio."""
        if not 0 <= replay_ratio < 1:
            raise ValueError("replay_ratio must be in [0, 1).")

        narrative_count = len(narrative_dataset)
        retention_count = int(
            math.ceil(
                narrative_count * replay_ratio / (1.0 - replay_ratio)
            )
        ) if replay_ratio > 0 else 0

        rng = np.random.default_rng(
            self.seed + round_index * 100_003 + epoch_index * 1_009
        )

        mixed_indices = list(range(narrative_count))

        if retention_count:
            retention_indices = rng.choice(
                len(retention_dataset),
                size=retention_count,
                replace=retention_count > len(retention_dataset),
            )
            mixed_indices.extend(
                narrative_count + int(index)
                for index in retention_indices
            )

        combined_dataset = ConcatDataset(
            [narrative_dataset, retention_dataset]
        )
        mixed_subset = Subset(combined_dataset, mixed_indices)
        generator = torch.Generator()
        generator.manual_seed(
            self.seed + round_index * 100_003 + epoch_index * 1_009
        )

        loader = DataLoader(
            mixed_subset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=self.collator,
            pin_memory=self.model_device.type == "cuda",
        )

        effective_ratio = retention_count / (narrative_count + retention_count)
        return loader, retention_count, effective_ratio

    def _autocast_context(self):
        if self.model_device.type == "cuda" and self.model_dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            return torch.autocast(
                device_type="cuda",
                dtype=self.model_dtype,
            )

        from contextlib import nullcontext
        return nullcontext()

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_supervised_tokens = 0

        with torch.inference_mode():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.model_device)
                attention_mask = batch["attention_mask"].to(self.model_device)
                labels = batch["labels"].to(self.model_device)

                supervised_tokens = int((labels[:, 1:] != -100).sum().item())
                if supervised_tokens == 0:
                    continue

                with self._autocast_context():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                total_loss += float(loss.item()) * supervised_tokens
                total_supervised_tokens += supervised_tokens

        if total_supervised_tokens == 0:
            raise RuntimeError("Evaluation data contains no supervised response tokens.")

        return total_loss / total_supervised_tokens

    def _train_round(
        self,
        narrative_dataset: SFTDataset,
        retention_dataset: SFTDataset,
        replay_ratio: float,
        round_index: int,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler,
        trainable_parameters: List[torch.nn.Parameter],
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        optimizer_steps = 0
        observed_replay_counts = []
        observed_ratios = []

        for epoch_index in range(self.epochs_per_round):
            loader, replay_count, effective_ratio = self._make_mixed_training_loader(
                narrative_dataset=narrative_dataset,
                retention_dataset=retention_dataset,
                replay_ratio=round_replay_ratio,
                round_index=round_index,
                epoch_index=epoch_index,
            )
            observed_replay_counts.append(replay_count)
            observed_ratios.append(effective_ratio)

            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulation_count = 0
            number_batches = len(loader)

            for batch_index, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(self.model_device)
                attention_mask = batch["attention_mask"].to(self.model_device)
                labels = batch["labels"].to(self.model_device)

                supervised_tokens = int((labels[:, 1:] != -100).sum().item())
                if supervised_tokens == 0:
                    continue

                with self._autocast_context():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError("Non-finite loss detected during continual SFT.")

                scaler.scale(
                    loss / self.gradient_accumulation_steps
                ).backward()

                accumulation_count += 1
                total_loss += float(loss.item()) * supervised_tokens
                total_tokens += supervised_tokens

                is_accumulation_step = (
                    accumulation_count == self.gradient_accumulation_steps
                )
                is_last_batch = batch_index == number_batches - 1

                if is_accumulation_step or is_last_batch:
                    if accumulation_count < self.gradient_accumulation_steps:
                        correction = (
                            self.gradient_accumulation_steps / accumulation_count
                        )
                        for parameter in trainable_parameters:
                            if parameter.grad is not None:
                                parameter.grad.mul_(correction)

                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=self.max_grad_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulation_count = 0
                    optimizer_steps += 1

        if total_tokens == 0:
            raise RuntimeError("The mixed training data contains no supervised response tokens.")

        return {
            "train_loss": total_loss / total_tokens,
            "optimizer_steps": float(optimizer_steps),
            "retention_examples_per_epoch": float(np.mean(observed_replay_counts)),
            "effective_replay_ratio": float(np.mean(observed_ratios)),
        }

    def fit(
        self,
        narrative_train_records: List[Dict],
        retention_train_records: List[Dict],
        narrative_eval_records: List[Dict],
        retention_eval_records: List[Dict],
    ) -> Dict:
        if min(
            len(narrative_train_records),
            len(retention_train_records),
            len(narrative_eval_records),
            len(retention_eval_records),
        ) < 2:
            raise ValueError("Each training and evaluation dataset must contain at least two examples.")

        narrative_train_dataset = self._make_dataset(narrative_train_records)
        retention_train_dataset = self._make_dataset(retention_train_records)
        narrative_eval_loader = self._make_eval_loader(
            self._make_dataset(narrative_eval_records)
        )
        retention_eval_loader = self._make_eval_loader(
            self._make_dataset(retention_eval_records)
        )

        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("No trainable LoRA parameters were found.")

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        max_replay_count = int(
            math.ceil(
                len(narrative_train_dataset)
                * self.max_replay_ratio
                / (1.0 - self.max_replay_ratio)
            )
        )
        max_examples_per_epoch = len(narrative_train_dataset) + max_replay_count
        max_batches_per_epoch = math.ceil(max_examples_per_epoch / self.batch_size)
        updates_per_epoch = math.ceil(
            max_batches_per_epoch / self.gradient_accumulation_steps
        )
        total_training_steps = max(
            1,
            self.max_rounds * self.epochs_per_round * updates_per_epoch,
        )
        warmup_steps = int(total_training_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_fp16_scaler
        )

        print("Evaluating the pretrained model before continual SFT.")
        baseline_narrative_loss = self.evaluate(narrative_eval_loader)
        baseline_retention_loss = self.evaluate(retention_eval_loader)

        if baseline_narrative_loss <= 0 or baseline_retention_loss <= 0:
            raise RuntimeError("Baseline evaluation losses must be positive.")

        self.history["baseline"] = {
            "narrative_loss": float(baseline_narrative_loss),
            "retention_loss": float(baseline_retention_loss),
        }

        replay_ratio = self.initial_replay_ratio
        previous_capability_deviation = None
        consecutive_converged_rounds = 0

        print("\nStarting continual SFT with adaptive skill replay.")
        print(f"Narrative training examples: {len(narrative_train_dataset)}")
        print(f"Retention training examples: {len(retention_train_dataset)}")
        print(f"Initial replay ratio: {replay_ratio:.3f}")
        print("-" * 95)

        for round_index in range(self.max_rounds):
            round_replay_ratio = replay_ratio
            round_training = self._train_round(
                narrative_dataset=narrative_train_dataset,
                retention_dataset=retention_train_dataset,
                replay_ratio=round_replay_ratio,
                round_index=round_index,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                trainable_parameters=trainable_parameters,
            )

            narrative_loss = self.evaluate(narrative_eval_loader)
            retention_loss = self.evaluate(retention_eval_loader)

            deviation = capability_deviation(
                baseline_retention_loss,
                retention_loss,
            )
            plasticity = plasticity_score(
                baseline_narrative_loss,
                narrative_loss,
            )
            stability = stability_score(
                previous_capability_deviation,
                deviation,
                self.stability_scale,
            )

            converged_this_round = (
                deviation <= self.max_capability_deviation
                and plasticity >= self.min_plasticity
                and stability >= self.min_stability
            )

            if converged_this_round:
                consecutive_converged_rounds += 1
            else:
                consecutive_converged_rounds = 0

            if consecutive_converged_rounds >= self.convergence_patience:
                ratio_update = "converged; replay ratio retained"
                self.history["converged"] = True
            elif deviation > self.max_capability_deviation:
                new_ratio = min(
                    self.max_replay_ratio,
                    replay_ratio + self.replay_ratio_step,
                )
                ratio_update = (
                    f"increased to {new_ratio:.3f}"
                    if new_ratio > replay_ratio
                    else "maximum replay ratio reached"
                )
                replay_ratio = new_ratio
            elif plasticity < self.min_plasticity:
                new_ratio = max(
                    self.min_replay_ratio,
                    replay_ratio - self.replay_ratio_step,
                )
                ratio_update = (
                    f"decreased to {new_ratio:.3f}"
                    if new_ratio < replay_ratio
                    else "minimum replay ratio reached"
                )
                replay_ratio = new_ratio
            else:
                ratio_update = "unchanged; checking stability"

            round_result = {
                "round": round_index + 1,
                "replay_ratio_used": float(round_replay_ratio),
                "next_replay_ratio": float(replay_ratio),
                "train_loss": float(round_training["train_loss"]),
                "narrative_eval_loss": float(narrative_loss),
                "retention_eval_loss": float(retention_loss),
                "capability_deviation": float(deviation),
                "stability": float(stability),
                "plasticity": float(plasticity),
                "retention_examples_per_epoch": int(
                    round_training["retention_examples_per_epoch"]
                ),
                "effective_replay_ratio": float(
                    round_training["effective_replay_ratio"]
                ),
                "optimizer_steps": int(round_training["optimizer_steps"]),
                "converged_this_round": bool(converged_this_round),
                "consecutive_converged_rounds": int(consecutive_converged_rounds),
                "ratio_update": ratio_update,
            }
            self.history["rounds"].append(round_result)
            self.history["final_replay_ratio"] = float(replay_ratio)

            print(
                f"Round [{round_index + 1:02d}/{self.max_rounds}] "
                f"| Replay: {round_result['replay_ratio_used']:.3f} "
                f"| Capability Deviation: {deviation:.5f} "
                f"| Stability: {stability:.5f} "
                f"| Plasticity: {plasticity:.5f}"
            )
            print(f"Replay adjustment: {ratio_update}")

            self.save_checkpoint()

            previous_capability_deviation = deviation
            if self.history["converged"]:
                break

        self.model.eval()
        self.model.config.use_cache = True
        self.save_checkpoint()
        self.save_final_adapter()

        if not self.history["converged"]:
            print("Maximum continual-learning rounds reached before convergence.")
        else:
            print("Continual-learning convergence criteria satisfied.")

        return self.history

    def save_checkpoint(self) -> None:
        history_path = os.path.join(
            self.output_dir,
            "continual_learning_history.json",
        )
        atomic_json_save(self.history, history_path)

    def save_final_adapter(self) -> None:
        adapter_dir = os.path.join(self.output_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)

        self.model.save_pretrained(
            adapter_dir,
            safe_serialization=True,
        )
        self.tokenizer.save_pretrained(adapter_dir)

        configuration = {
            "base_model": self.model_name_or_path,
            "fine_tuning_method": "LoRA",
            "continual_learning_method": "adaptive_replay_during_sft",
            "narrative_dataset_role": "new_domain_adaptation",
            "retention_dataset_role": "pretrained_capability_replay",
            "replay_ratio_definition": "fraction of mixed training examples drawn from retention data",
            "initial_replay_ratio": self.initial_replay_ratio,
            "min_replay_ratio": self.min_replay_ratio,
            "max_replay_ratio": self.max_replay_ratio,
            "replay_ratio_step": self.replay_ratio_step,
            "max_capability_deviation": self.max_capability_deviation,
            "min_plasticity": self.min_plasticity,
            "min_stability": self.min_stability,
            "stability_scale": self.stability_scale,
            "convergence_patience": self.convergence_patience,
            "max_seq_length": self.max_seq_length,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "epochs_per_round": self.epochs_per_round,
            "max_rounds": self.max_rounds,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "precision": str(self.model_dtype),
            "load_in_4bit": self.load_in_4bit,
            "gradient_checkpointing": self.gradient_checkpointing,
            "baseline_metrics": self.history["baseline"],
            "converged": self.history["converged"],
            "final_replay_ratio": self.history["final_replay_ratio"],
            "metric_definitions": {
                "capability_deviation": "max(0, (current retention loss - baseline retention loss) / baseline retention loss); lower is better",
                "stability": "1 - min(abs(current capability deviation - previous capability deviation) / stability_scale, 1); higher is better",
                "plasticity": "(baseline narrative loss - current narrative loss) / baseline narrative loss; higher is better",
            },
        }
        atomic_json_save(
            configuration,
            os.path.join(self.output_dir, "continual_learning_config.json"),
        )

        print(f"Continual-learning adapter saved to: {adapter_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Continual SFT for Yi-6B-200K using adaptive replay of pretrained skills."
        )
    )

    parser.add_argument("--model-name-or-path", type=str, default="./models/Yi-6B-200K")
    parser.add_argument("--narrative-train", type=str, required=True)
    parser.add_argument("--retention-train", type=str, required=True)
    parser.add_argument("--narrative-eval", type=str, required=True)
    parser.add_argument("--retention-eval", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/continual_learning")

    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs-per-round", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--initial-replay-ratio", type=float, default=0.25)
    parser.add_argument("--min-replay-ratio", type=float, default=0.05)
    parser.add_argument("--max-replay-ratio", type=float, default=0.80)
    parser.add_argument("--replay-ratio-step", type=float, default=0.05)
    parser.add_argument("--max-capability-deviation", type=float, default=0.05)
    parser.add_argument("--min-plasticity", type=float, default=0.05)
    parser.add_argument("--min-stability", type=float, default=0.90)
    parser.add_argument("--stability-scale", type=float, default=0.05)
    parser.add_argument("--convergence-patience", type=int, default=2)

    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument(
        "--precision",
        type=str,
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    narrative_train_records = load_jsonl(args.narrative_train)
    retention_train_records = load_jsonl(args.retention_train)
    narrative_eval_records = load_jsonl(args.narrative_eval)
    retention_eval_records = load_jsonl(args.retention_eval)

    target_modules = [
        module.strip()
        for module in args.lora_target_modules.split(",")
        if module.strip()
    ]
    if not target_modules:
        raise ValueError("At least one LoRA target module must be specified.")

    trainer = ContinualLearningTrainer(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs_per_round=args.epochs_per_round,
        max_rounds=args.max_rounds,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        initial_replay_ratio=args.initial_replay_ratio,
        min_replay_ratio=args.min_replay_ratio,
        max_replay_ratio=args.max_replay_ratio,
        replay_ratio_step=args.replay_ratio_step,
        max_capability_deviation=args.max_capability_deviation,
        min_plasticity=args.min_plasticity,
        min_stability=args.min_stability,
        stability_scale=args.stability_scale,
        convergence_patience=args.convergence_patience,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=target_modules,
        precision=args.precision,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        device=args.device,
        seed=args.seed,
        prompt_template=args.prompt_template,
    )

    history = trainer.fit(
        narrative_train_records=narrative_train_records,
        retention_train_records=retention_train_records,
        narrative_eval_records=narrative_eval_records,
        retention_eval_records=retention_eval_records,
    )

    print("\nContinual-learning summary")
    print(f"Converged: {history['converged']}")
    print(f"Final replay ratio: {history['final_replay_ratio']:.3f}")
    print(f"Rounds completed: {len(history['rounds'])}")


if __name__ == "__main__":
    main()
