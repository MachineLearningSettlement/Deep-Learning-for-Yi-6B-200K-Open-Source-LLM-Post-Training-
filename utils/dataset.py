"""Reusable JSONL dataset utilities for Yi-6B-200K post-training."""

import json
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def validate_prompt_response_record(record: Dict, line_number: Optional[int] = None) -> Dict:
    """Validate and return one prompt-response record."""
    location = f" at line {line_number}" if line_number is not None else ""

    if not isinstance(record, dict):
        raise ValueError(f"Each dataset entry must be a JSON object{location}.")

    if "prompt" not in record or "response" not in record:
        raise ValueError(f"Each record must contain 'prompt' and 'response'{location}.")

    if not isinstance(record["prompt"], str) or not record["prompt"].strip():
        raise ValueError(f"The prompt must be a non-empty string{location}.")

    if not isinstance(record["response"], str) or not record["response"].strip():
        raise ValueError(f"The response must be a non-empty string{location}.")

    return record


def load_curated_indices(path: str, dataset_size: int) -> np.ndarray:
    """Load and validate sample indices produced by dataset_curator.py."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Curated indices file not found: {path}")

    indices = np.asarray(np.load(path, allow_pickle=False))

    if indices.ndim != 1:
        raise ValueError("Curated indices must be a one-dimensional array.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Curated indices must contain integers.")
    if indices.size == 0:
        raise ValueError("The curated index array is empty.")
    if np.unique(indices).size != indices.size:
        raise ValueError("Curated indices must be unique.")
    if np.any(indices < 0) or np.any(indices >= dataset_size):
        raise IndexError("Curated indices exceed the original dataset boundaries.")

    return indices.astype(np.int64, copy=False)


def load_jsonl(
    path: str,
    curated_indices: Optional[str] = None,
    minimum_records: int = 2,
) -> List[Dict]:
    """Load prompt-response records from JSONL, optionally applying curated indices."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    if minimum_records < 1:
        raise ValueError("minimum_records must be at least 1.")

    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number}: {error}") from error
            records.append(validate_prompt_response_record(record, line_number))

    if len(records) < minimum_records:
        raise ValueError(
            f"The dataset contains {len(records)} valid records; "
            f"at least {minimum_records} are required."
        )

    if curated_indices is not None:
        indices = load_curated_indices(curated_indices, len(records))
        records = [records[int(index)] for index in indices]
        if len(records) < minimum_records:
            raise ValueError(
                f"The curated dataset contains {len(records)} records; "
                f"at least {minimum_records} are required."
            )

    return records


def write_jsonl(records: Iterable[Dict], path: str) -> None:
    """Validate and write prompt-response records as UTF-8 JSONL."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    temporary_path = f"{path}.tmp"
    count = 0
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            for count, record in enumerate(records, start=1):
                validate_prompt_response_record(record, count)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if count == 0:
            raise ValueError("Cannot write an empty dataset.")
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def split_records(
    records: Sequence[Dict],
    validation_split: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Create a reproducible shuffled train/validation split."""
    if len(records) < 2:
        raise ValueError("At least two records are required for a train/validation split.")
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be strictly between 0 and 1.")

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    validation_size = min(len(records) - 1, max(1, int(round(len(records) * validation_split))))

    validation_indices = set(indices[:validation_size])
    train_records = [record for index, record in enumerate(records) if index not in validation_indices]
    validation_records = [record for index, record in enumerate(records) if index in validation_indices]

    return train_records, validation_records
