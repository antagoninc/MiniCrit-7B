"""Data loading and preprocessing for MiniCrit training.

This module handles loading CSV data, tokenization with proper label masking,
data deduplication, validation splits, and quality validation.

Example:
    >>> from src.data import load_and_prepare_data, tokenize_dataset
    >>> df = load_and_prepare_data("data.csv", sample_size=1000)
    >>> dataset = tokenize_dataset(df, tokenizer, "text", "rebuttal", 512)

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
from typing import TYPE_CHECKING

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Minimum rebuttal word count per CLAUDE.md
MIN_REBUTTAL_WORDS = 50


def find_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Find text and rebuttal columns in the dataframe.

    Searches for common column name patterns for input text and output
    rebuttal/critique columns.

    Args:
        df: Input dataframe to search for columns.

    Returns:
        Tuple of (text_column_name, rebuttal_column_name).

    Raises:
        ValueError: If required columns are not found.

    Example:
        >>> df = pd.DataFrame({"rationale": ["..."], "critique": ["..."]})
        >>> text_col, rebuttal_col = find_columns(df)
        >>> print(text_col, rebuttal_col)
        'rationale' 'critique'
    """
    text_candidates = ["text", "rationale", "input", "prompt"]
    rebuttal_candidates = ["rebuttal", "critique", "response", "output"]

    text_col = None
    for col in text_candidates:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            f"No text column found. Tried: {text_candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )

    rebuttal_col = None
    for col in rebuttal_candidates:
        if col in df.columns:
            rebuttal_col = col
            break

    if rebuttal_col is None:
        raise ValueError(
            f"No rebuttal column found. Tried: {rebuttal_candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return text_col, rebuttal_col


def compute_text_hash(text: str) -> str:
    """Compute a hash for text deduplication.

    Args:
        text: Input text string.

    Returns:
        MD5 hash of normalized text.
    """
    # Normalize: lowercase, strip whitespace, remove extra spaces
    normalized = " ".join(str(text).lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def deduplicate_data(
    df: pd.DataFrame,
    text_col: str,
    rebuttal_col: str,
    keep: str = "first",
) -> pd.DataFrame:
    """Remove duplicate and near-duplicate examples from the dataset.

    Deduplicates based on both the input text and rebuttal content to
    ensure training diversity.

    Args:
        df: Input dataframe.
        text_col: Name of the text column.
        rebuttal_col: Name of the rebuttal column.
        keep: Which duplicate to keep ('first', 'last', or False to drop all).

    Returns:
        Deduplicated dataframe.

    Example:
        >>> df_deduped = deduplicate_data(df, "text", "rebuttal")
        >>> print(f"Removed {len(df) - len(df_deduped)} duplicates")
    """
    initial_len = len(df)

    # Create hash columns for deduplication
    df = df.copy()
    df["_text_hash"] = df[text_col].apply(compute_text_hash)
    df["_rebuttal_hash"] = df[rebuttal_col].apply(compute_text_hash)
    df["_combined_hash"] = df["_text_hash"] + df["_rebuttal_hash"]

    # Remove exact duplicates (same text + same rebuttal)
    df = df.drop_duplicates(subset=["_combined_hash"], keep=keep)

    # Also remove cases where the same input has different rebuttals
    # (keeps first rebuttal for each unique input)
    df = df.drop_duplicates(subset=["_text_hash"], keep=keep)

    # Clean up temporary columns
    df = df.drop(columns=["_text_hash", "_rebuttal_hash", "_combined_hash"])

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Deduplication removed {removed:,} duplicate rows")

    return df.reset_index(drop=True)


def validate_rebuttal_length(
    df: pd.DataFrame,
    rebuttal_col: str,
    min_words: int = MIN_REBUTTAL_WORDS,
) -> pd.DataFrame:
    """Filter out rebuttals that are too short.

    Per CLAUDE.md: "Do NOT generate rebuttals shorter than 50 words -
    they lack substance."

    Args:
        df: Input dataframe.
        rebuttal_col: Name of the rebuttal column.
        min_words: Minimum word count for rebuttals.

    Returns:
        Filtered dataframe with only substantial rebuttals.

    Example:
        >>> df_filtered = validate_rebuttal_length(df, "rebuttal", min_words=50)
    """
    initial_len = len(df)

    # Count words in each rebuttal
    word_counts = df[rebuttal_col].apply(lambda x: len(str(x).split()))

    # Filter
    df = df[word_counts >= min_words].copy()

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(
            f"Removed {removed:,} rows with rebuttals < {min_words} words "
            f"(per CLAUDE.md requirements)"
        )

    return df.reset_index(drop=True)


def load_and_prepare_data(
    data_file: str,
    sample_size: int | None = None,
    min_text_length: int = 10,
    min_rebuttal_words: int = MIN_REBUTTAL_WORDS,
    deduplicate: bool = True,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, str, str]:
    """Load and clean the training data from CSV.

    Loads the CSV file, identifies text and rebuttal columns, removes
    null values, short texts, duplicates, and rebuttals that lack substance.

    Args:
        data_file: Path to the CSV file containing training data.
        sample_size: If provided, randomly sample this many rows. Useful
            for testing and debugging.
        min_text_length: Minimum character length for text column.
        min_rebuttal_words: Minimum word count for rebuttals (default 50
            per CLAUDE.md requirements).
        deduplicate: Whether to remove duplicate examples.
        random_seed: Random seed for reproducible sampling.

    Returns:
        Tuple of (cleaned_dataframe, text_column_name, rebuttal_column_name).

    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If required columns are not found.

    Example:
        >>> df, text_col, rebuttal_col = load_and_prepare_data(
        ...     "train.csv", sample_size=1000
        ... )
        >>> print(f"Loaded {len(df)} rows")
    """
    logger.info(f"Loading CSV: {data_file}")
    df = pd.read_csv(data_file, low_memory=False)
    logger.info(f"Raw rows: {len(df):,}")

    if sample_size is not None:
        df = df.sample(sample_size, random_state=random_seed).reset_index(drop=True)
        logger.warning(f"SAMPLED to {len(df):,} rows")

    text_col, rebuttal_col = find_columns(df)
    logger.info(f"Using columns: {text_col} -> {rebuttal_col}")

    # Clean data - remove nulls
    initial_len = len(df)
    df = df.dropna(subset=[text_col, rebuttal_col])
    null_removed = initial_len - len(df)
    if null_removed > 0:
        logger.info(f"Removed {null_removed:,} rows with null values")

    # Remove short texts
    initial_len = len(df)
    df = df[df[text_col].str.len() > min_text_length]
    short_removed = initial_len - len(df)
    if short_removed > 0:
        logger.info(f"Removed {short_removed:,} rows with text < {min_text_length} chars")

    # Validate rebuttal length (per CLAUDE.md requirements)
    df = validate_rebuttal_length(df, rebuttal_col, min_rebuttal_words)

    # Deduplicate
    if deduplicate:
        df = deduplicate_data(df, text_col, rebuttal_col)

    logger.info(f"After cleaning: {len(df):,} rows")

    return df, text_col, rebuttal_col


def create_train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.05,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into training and validation sets.

    Creates a stratified-like split ensuring validation set is representative.

    Args:
        df: Input dataframe.
        val_ratio: Fraction of data to use for validation (default 5%).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df).

    Example:
        >>> train_df, val_df = create_train_val_split(df, val_ratio=0.05)
        >>> print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    """
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Calculate split point
    val_size = int(len(df) * val_ratio)
    val_size = max(val_size, 100)  # Ensure at least 100 validation examples

    val_df = df.iloc[:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)

    logger.info(f"Split data: {len(train_df):,} train, {len(val_df):,} validation")

    return train_df, val_df


def create_tokenize_function(
    tokenizer: PreTrainedTokenizer,
    text_col: str,
    rebuttal_col: str,
    max_length: int,
) -> callable:
    """Create a tokenization function with proper label masking.

    Creates a function that tokenizes input-output pairs, masking the
    input tokens in labels so only the output (critique) is trained on.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        text_col: Name of the input text column.
        rebuttal_col: Name of the output rebuttal column.
        max_length: Maximum sequence length.

    Returns:
        Tokenization function compatible with datasets.map().

    Note:
        The returned function masks input tokens by setting their labels
        to -100, which is ignored by the loss function. This ensures the
        model is only trained to generate critiques, not to repeat inputs.
    """
    def tokenize_with_masked_labels(
        examples: dict[str, list[str]],
    ) -> dict[str, list[list[int]]]:
        """Tokenize examples with masked labels for prompt tokens.

        Args:
            examples: Batch of examples with text and rebuttal columns.

        Returns:
            Dictionary with input_ids, attention_mask, and labels.
        """
        all_input_ids: list[list[int]] = []
        all_attention_mask: list[list[int]] = []
        all_labels: list[list[int]] = []

        for text, rebuttal in zip(examples[text_col], examples[rebuttal_col]):
            prompt = f"### Rationale:\n{text}\n\n### Critique:\n"
            response = str(rebuttal) + tokenizer.eos_token

            prompt_tokens = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length - 50,
            )

            response_tokens = tokenizer(
                response,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length - len(prompt_tokens["input_ids"]),
            )

            input_ids = prompt_tokens["input_ids"] + response_tokens["input_ids"]
            attention_mask = [1] * len(input_ids)
            # Mask prompt tokens with -100 so they're ignored in loss
            labels = [-100] * len(prompt_tokens["input_ids"]) + response_tokens["input_ids"]

            # Pad or truncate to max_length
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                labels = labels + [-100] * pad_len
            else:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    return tokenize_with_masked_labels


def tokenize_dataframe(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    text_col: str,
    rebuttal_col: str,
    max_length: int,
    chunk_size: int = 500000,
) -> Dataset:
    """Tokenize a dataframe into a HuggingFace Dataset.

    Internal function that handles chunked tokenization for memory efficiency.

    Args:
        df: Input dataframe.
        tokenizer: Tokenizer instance.
        text_col: Text column name.
        rebuttal_col: Rebuttal column name.
        max_length: Maximum sequence length.
        chunk_size: Rows per processing chunk.

    Returns:
        Tokenized Dataset.
    """
    tokenize_fn = create_tokenize_function(tokenizer, text_col, rebuttal_col, max_length)

    all_chunks: list[Dataset] = []
    total_rows = len(df)

    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        logger.info(f"Processing chunk {start_idx:,} - {end_idx:,} / {total_rows:,}")

        chunk_df = df.iloc[start_idx:end_idx]
        chunk_dataset = Dataset.from_pandas(
            chunk_df[[text_col, rebuttal_col]].reset_index(drop=True)
        )

        chunk_tok = chunk_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=500,
            num_proc=1,
            remove_columns=chunk_dataset.column_names,
            desc=f"TOKENIZE {start_idx // chunk_size + 1}",
        )
        all_chunks.append(chunk_tok)

        del chunk_df, chunk_dataset
        gc.collect()

    logger.info("Concatenating chunks...")
    dataset_tok = concatenate_datasets(all_chunks)
    del all_chunks
    gc.collect()

    return dataset_tok


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    text_col: str,
    rebuttal_col: str,
    max_length: int,
    cache_dir: str | None = None,
    use_cache: bool = True,
    chunk_size: int = 500000,
) -> Dataset:
    """Tokenize the dataset with proper label masking.

    Processes the dataframe in chunks to avoid memory issues, applies
    tokenization with label masking, and optionally caches the result.

    Args:
        df: Input dataframe with text and rebuttal columns.
        tokenizer: HuggingFace tokenizer instance.
        text_col: Name of the text/rationale column.
        rebuttal_col: Name of the rebuttal/critique column.
        max_length: Maximum sequence length for tokenization.
        cache_dir: Directory to cache tokenized dataset. If None, no caching.
        use_cache: Whether to use/save cache. If True and cache exists,
            loads from cache instead of re-tokenizing.
        chunk_size: Number of rows to process per chunk. Lower values
            use less memory but are slower.

    Returns:
        Tokenized HuggingFace Dataset ready for training.

    Example:
        >>> dataset = tokenize_dataset(
        ...     df, tokenizer, "text", "rebuttal", 512,
        ...     cache_dir="./cache", use_cache=True
        ... )
    """
    # Try to load from cache first
    if use_cache and cache_dir and os.path.exists(cache_dir):
        logger.info(f"Loading cached tokenized dataset from {cache_dir}...")
        dataset_tok = load_from_disk(cache_dir)
        logger.info(f"Loaded {len(dataset_tok):,} examples from cache")
        return dataset_tok

    logger.info("Tokenizing dataset with proper label masking...")
    logger.info("(Only critique/rebuttal tokens will be trained on)")

    dataset_tok = tokenize_dataframe(
        df, tokenizer, text_col, rebuttal_col, max_length, chunk_size
    )

    logger.info(f"Tokenized {len(dataset_tok):,} examples")

    # Save to cache
    if use_cache and cache_dir:
        logger.info(f"Saving cache to {cache_dir}...")
        dataset_tok.save_to_disk(cache_dir)
        logger.info("Cache saved successfully")

    return dataset_tok


def tokenize_dataset_with_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    text_col: str,
    rebuttal_col: str,
    max_length: int,
    cache_dir: str | None = None,
    use_cache: bool = True,
) -> DatasetDict:
    """Tokenize train and validation datasets.

    Args:
        train_df: Training dataframe.
        val_df: Validation dataframe.
        tokenizer: Tokenizer instance.
        text_col: Text column name.
        rebuttal_col: Rebuttal column name.
        max_length: Maximum sequence length.
        cache_dir: Cache directory for tokenized data.
        use_cache: Whether to use caching.

    Returns:
        DatasetDict with 'train' and 'validation' splits.

    Example:
        >>> datasets = tokenize_dataset_with_split(
        ...     train_df, val_df, tokenizer, "text", "rebuttal", 512
        ... )
        >>> train_data = datasets["train"]
        >>> val_data = datasets["validation"]
    """
    # Check cache
    if use_cache and cache_dir:
        train_cache = os.path.join(cache_dir, "train")
        val_cache = os.path.join(cache_dir, "validation")

        if os.path.exists(train_cache) and os.path.exists(val_cache):
            logger.info(f"Loading cached datasets from {cache_dir}...")
            train_tok = load_from_disk(train_cache)
            val_tok = load_from_disk(val_cache)
            logger.info(
                f"Loaded {len(train_tok):,} train, {len(val_tok):,} validation examples"
            )
            return DatasetDict({"train": train_tok, "validation": val_tok})

    logger.info("Tokenizing training data...")
    train_tok = tokenize_dataframe(
        train_df, tokenizer, text_col, rebuttal_col, max_length
    )

    logger.info("Tokenizing validation data...")
    val_tok = tokenize_dataframe(
        val_df, tokenizer, text_col, rebuttal_col, max_length
    )

    logger.info(
        f"Tokenized {len(train_tok):,} train, {len(val_tok):,} validation examples"
    )

    # Save to cache
    if use_cache and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        train_cache = os.path.join(cache_dir, "train")
        val_cache = os.path.join(cache_dir, "validation")

        logger.info(f"Saving cache to {cache_dir}...")
        train_tok.save_to_disk(train_cache)
        val_tok.save_to_disk(val_cache)
        logger.info("Cache saved successfully")

    return DatasetDict({"train": train_tok, "validation": val_tok})


def validate_dataset(
    dataset: Dataset,
    check_count: int = 100,
    min_valid_ratio: float = 0.9,
) -> bool:
    """Validate the tokenized dataset for training readiness.

    Checks that examples have trainable tokens (labels that aren't -100)
    and that the dataset meets minimum quality thresholds.

    Args:
        dataset: Tokenized dataset to validate.
        check_count: Number of examples to check (from the beginning).
        min_valid_ratio: Minimum ratio of valid examples required.
            Default 0.9 means at least 90% must have trainable tokens.

    Returns:
        True if validation passed, False otherwise.

    Example:
        >>> is_valid = validate_dataset(tokenized_data)
        >>> if not is_valid:
        ...     raise ValueError("Dataset validation failed")
    """
    logger.info("Validating dataset...")

    # Check first example in detail
    sample = dataset[0]
    total_tokens = len(sample["labels"])
    trainable_tokens = sum(1 for label in sample["labels"] if label != -100)
    masked_tokens = total_tokens - trainable_tokens

    logger.info(
        f"Example 0: total={total_tokens}, trainable={trainable_tokens}, "
        f"masked={masked_tokens}"
    )

    if trainable_tokens == 0:
        logger.error("No trainable tokens! All labels are -100.")
        return False

    if trainable_tokens < 10:
        logger.warning(f"Only {trainable_tokens} trainable tokens in first example.")

    # Check multiple examples
    actual_check_count = min(check_count, len(dataset))
    valid_count = 0

    for i in range(actual_check_count):
        trainable = sum(1 for label in dataset[i]["labels"] if label != -100)
        if trainable > 0:
            valid_count += 1

    valid_ratio = valid_count / actual_check_count
    logger.info(
        f"Valid examples (first {actual_check_count}): "
        f"{valid_count}/{actual_check_count} ({valid_ratio:.1%})"
    )

    if valid_ratio < min_valid_ratio:
        logger.error(
            f"Too many examples have no trainable tokens! "
            f"Valid ratio {valid_ratio:.1%} < required {min_valid_ratio:.1%}"
        )
        return False

    logger.info("Dataset validation PASSED")
    return True
