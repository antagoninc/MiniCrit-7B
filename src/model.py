"""Model loading and configuration for MiniCrit training.

This module handles loading the base model, tokenizer configuration,
and applying LoRA adapters.

Example:
    >>> from src.model import load_model_and_tokenizer, apply_lora
    >>> from src.config import load_config
    >>> config = load_config()
    >>> model, tokenizer = load_model_and_tokenizer(config.model_name)
    >>> model = apply_lora(model, config.lora)

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel

    from src.config import LoRAConfig

logger = logging.getLogger(__name__)


def check_gpu() -> dict[str, str | float] | None:
    """Check GPU availability and return details.

    Checks for CUDA availability and returns GPU information if found.

    Returns:
        Dictionary with 'name' and 'memory_gb' keys if GPU is available,
        None otherwise.

    Example:
        >>> gpu_info = check_gpu()
        >>> if gpu_info:
        ...     print(f"Using {gpu_info['name']} with {gpu_info['memory_gb']}GB")
    """
    logger.info("Checking GPU availability...")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"VRAM: {gpu_memory:.1f} GB")
        return {"name": gpu_name, "memory_gb": gpu_memory}
    else:
        logger.warning("No GPU found!")
        return None


def require_gpu() -> None:
    """Require GPU and exit if not available.

    Checks for GPU availability and exits the program if no GPU is found.
    Use this at the start of training scripts that require a GPU.

    Raises:
        SystemExit: If no GPU is available.
    """
    if check_gpu() is None:
        logger.error("Training requires CUDA GPU. Exiting.")
        sys.exit(1)


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Load and configure the tokenizer.

    Loads the tokenizer from HuggingFace and configures padding settings.
    If no pad token is defined, uses the EOS token as padding.

    Args:
        model_name: HuggingFace model identifier or local path.

    Returns:
        Configured tokenizer instance ready for use.

    Example:
        >>> tokenizer = load_tokenizer("Qwen/Qwen2-7B-Instruct")
        >>> print(f"Vocab size: {tokenizer.vocab_size}")
    """
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    tokenizer.padding_side = "right"

    logger.info(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    logger.info(
        f"pad_token_id: {tokenizer.pad_token_id}, "
        f"eos_token_id: {tokenizer.eos_token_id}"
    )

    return tokenizer


def load_base_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "eager",
) -> PreTrainedModel:
    """Load the base language model.

    Loads a causal language model from HuggingFace with memory-efficient
    settings and enables gradient checkpointing.

    Args:
        model_name: HuggingFace model identifier or local path.
        torch_dtype: PyTorch data type for model weights. Default bfloat16
            provides good balance of precision and memory usage.
        device_map: Device placement strategy. "auto" automatically
            distributes across available devices.
        attn_implementation: Attention implementation to use. Options include
            "eager", "sdpa", "flash_attention_2".

    Returns:
        Loaded model with gradient checkpointing enabled.

    Example:
        >>> model = load_base_model("Qwen/Qwen2-7B-Instruct")
        >>> print(f"Parameters: {model.num_parameters():,}")
    """
    logger.info(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )

    model.gradient_checkpointing_enable()
    logger.info("Model loaded with gradient checkpointing enabled")

    return model


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load both model and tokenizer.

    Convenience function to load both the model and tokenizer in one call.

    Args:
        model_name: HuggingFace model identifier or local path.
        torch_dtype: PyTorch data type for model weights.

    Returns:
        Tuple of (model, tokenizer).

    Example:
        >>> model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2-7B-Instruct")
    """
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, torch_dtype=torch_dtype)
    return model, tokenizer


def apply_lora(
    model: PreTrainedModel,
    lora_config: LoRAConfig,
) -> PeftModel:
    """Apply LoRA adapters to the model.

    Wraps the base model with LoRA (Low-Rank Adaptation) adapters for
    parameter-efficient fine-tuning.

    Args:
        model: Base language model to apply LoRA to.
        lora_config: LoRA configuration parameters.

    Returns:
        Model wrapped with LoRA adapters.

    Note:
        After applying LoRA, only a small fraction of parameters are
        trainable (typically ~0.5% of total parameters).

    Example:
        >>> from src.config import LoRAConfig
        >>> lora_config = LoRAConfig(r=16, alpha=32)
        >>> model = apply_lora(base_model, lora_config)
        >>> model.print_trainable_parameters()
    """
    logger.info("Applying LoRA adapters...")

    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
    )

    model = get_peft_model(model, peft_config)

    # Log trainable parameters
    model.print_trainable_parameters()

    return model


def save_model(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    output_path: str,
) -> None:
    """Save the fine-tuned model and tokenizer.

    Saves the LoRA adapter weights and tokenizer to the specified path.

    Args:
        model: Fine-tuned model with LoRA adapters.
        tokenizer: Tokenizer instance.
        output_path: Directory path to save the model.

    Example:
        >>> save_model(model, tokenizer, "checkpoints/final")
    """
    logger.info(f"Saving model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Model and tokenizer saved successfully")
