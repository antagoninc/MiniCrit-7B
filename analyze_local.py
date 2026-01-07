#!/usr/bin/env python3
"""Analyze MiniCrit training checkpoints and generate visualizations.

This script parses trainer state from checkpoints and generates training
curve visualizations including loss, learning rate, and gradient norms.

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3

Example usage:
    # Analyze default checkpoint directory
    python analyze_local.py

    # Analyze specific directory
    python analyze_local.py --checkpoint-dir path/to/checkpoints

    # Save to custom output path
    python analyze_local.py --output results/curves.png

    # Customize smoothing window
    python analyze_local.py --smoothing-window 50

    # Skip display (useful for headless servers)
    python analyze_local.py --no-display
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Analyze MiniCrit training checkpoints and generate visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        "-c",
        type=str,
        default="minicrit_7b_output",
        help="Directory containing training checkpoints",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="training_curves.png",
        help="Output path for the training curves image",
    )
    parser.add_argument(
        "--smoothing-window",
        "-w",
        type=int,
        default=20,
        help="Window size for smoothing the loss curve",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI (resolution) for the output image",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the plot (useful for headless servers)",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="14,10",
        help="Figure size as 'width,height' in inches",
    )
    parser.add_argument(
        "--json-output",
        "-j",
        type=str,
        default=None,
        help="Optional: Export metrics summary to JSON file",
    )
    return parser.parse_args()


def find_checkpoints(checkpoint_dir: str) -> list[Path]:
    """Find all checkpoint directories in the given path.

    Args:
        checkpoint_dir: Path to search for checkpoints.

    Returns:
        List of checkpoint paths sorted by step number.
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return []

    checkpoints = [
        d for d in checkpoint_path.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))

    return checkpoints


def load_trainer_state(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load trainer state from a checkpoint directory.

    Args:
        checkpoint_path: Path to the checkpoint directory.

    Returns:
        Trainer state dictionary, or None if not found.
    """
    state_file = checkpoint_path / "trainer_state.json"

    if not state_file.exists():
        logger.error(f"Trainer state not found: {state_file}")
        return None

    with open(state_file, "r") as f:
        return json.load(f)


def extract_metrics(log_history: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Extract training metrics from log history.

    Args:
        log_history: List of log entries from trainer state.

    Returns:
        Dictionary with lists of steps, losses, learning_rates, and grad_norms.
    """
    steps: list[int] = []
    losses: list[float] = []
    learning_rates: list[float] = []
    grad_norms: list[float] = []

    for entry in log_history:
        if "loss" in entry:
            steps.append(entry.get("step", 0))
            losses.append(entry["loss"])
            learning_rates.append(entry.get("learning_rate", 0))
            grad_norms.append(entry.get("grad_norm", 0))

    return {
        "steps": steps,
        "losses": losses,
        "learning_rates": learning_rates,
        "grad_norms": grad_norms,
    }


def smooth_values(values: list[float], window: int) -> list[float]:
    """Apply moving average smoothing to values.

    Args:
        values: List of values to smooth.
        window: Window size for moving average.

    Returns:
        List of smoothed values.
    """
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def compute_summary(
    state: dict[str, Any],
    metrics: dict[str, list[float]],
) -> dict[str, Any]:
    """Compute summary statistics from training metrics.

    Args:
        state: Trainer state dictionary.
        metrics: Extracted metrics dictionary.

    Returns:
        Summary statistics dictionary.
    """
    losses = metrics["losses"]
    grad_norms = metrics["grad_norms"]

    valid_grads = [g for g in grad_norms if g > 0]

    summary = {
        "steps_completed": state.get("global_step", 0),
        "steps_planned": state.get("max_steps", 0),
        "progress_percent": (
            state.get("global_step", 0) / max(state.get("max_steps", 1), 1) * 100
        ),
        "initial_loss": losses[0] if losses else 0,
        "final_loss": losses[-1] if losses else 0,
        "loss_reduction_percent": (
            (1 - losses[-1] / losses[0]) * 100 if losses and losses[0] > 0 else 0
        ),
        "avg_grad_norm": sum(valid_grads) / len(valid_grads) if valid_grads else 0,
        "min_loss": min(losses) if losses else 0,
        "max_loss": max(losses) if losses else 0,
    }

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print training summary to console.

    Args:
        summary: Summary statistics dictionary.
    """
    print("\n" + "=" * 60)
    print("MiniCrit-7B Training Analysis")
    print("Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3")
    print("=" * 60)

    print(f"\nTRAINING SUMMARY")
    print(f"   Steps completed: {summary['steps_completed']:,}")
    print(f"   Steps planned: {summary['steps_planned']:,}")
    print(f"   Progress: {summary['progress_percent']:.1f}%")

    print(f"\nLOSS METRICS")
    print(f"   Initial loss: {summary['initial_loss']:.4f}")
    print(f"   Final loss: {summary['final_loss']:.4f}")
    print(f"   Min loss: {summary['min_loss']:.4f}")
    print(f"   Reduction: {summary['loss_reduction_percent']:.1f}%")

    print(f"\nGRADIENT METRICS")
    print(f"   Avg grad norm: {summary['avg_grad_norm']:.4f}")

    print(f"\nHardware: NVIDIA H100 (Lambda Labs GPU Grant)")


def create_visualization(
    metrics: dict[str, list[float]],
    output_path: str,
    smoothing_window: int = 20,
    figsize: tuple[int, int] = (14, 10),
    dpi: int = 150,
    show: bool = True,
) -> None:
    """Create and save training visualization.

    Args:
        metrics: Dictionary of training metrics.
        output_path: Path to save the figure.
        smoothing_window: Window size for loss smoothing.
        figsize: Figure size in inches (width, height).
        dpi: Figure resolution.
        show: Whether to display the plot.
    """
    steps = metrics["steps"]
    losses = metrics["losses"]
    learning_rates = metrics["learning_rates"]
    grad_norms = metrics["grad_norms"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "MiniCrit-7B Training Analysis\n"
        "Antagon Inc. | Trained with Lambda Labs GPU Grant",
        fontsize=14,
        fontweight="bold",
    )

    # Loss curve (raw)
    ax1 = axes[0, 0]
    ax1.plot(steps, losses, "b-", linewidth=0.8, alpha=0.7)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Over Time")
    ax1.grid(True, alpha=0.3)
    if losses:
        ax1.set_ylim(0, max(losses) * 1.1)

    # Loss curve (smoothed)
    ax2 = axes[0, 1]
    smoothed = smooth_values(losses, smoothing_window)
    ax2.plot(steps, smoothed, "b-", linewidth=1.5)
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Loss (Smoothed)")
    ax2.set_title(f"Training Loss (Smoothed, window={smoothing_window})")
    ax2.grid(True, alpha=0.3)

    # Learning rate
    ax3 = axes[1, 0]
    ax3.plot(steps, learning_rates, "g-", linewidth=0.8)
    ax3.set_xlabel("Training Steps")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule (Cosine)")
    ax3.grid(True, alpha=0.3)

    # Gradient norm
    ax4 = axes[1, 1]
    ax4.plot(steps, grad_norms, "r-", linewidth=0.8, alpha=0.7)
    ax4.set_xlabel("Training Steps")
    ax4.set_ylabel("Gradient Norm")
    ax4.set_title("Gradient Norm Over Time")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved visualization: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def analyze_training(
    checkpoint_dir: str,
    output_path: str,
    smoothing_window: int = 20,
    figsize: tuple[int, int] = (14, 10),
    dpi: int = 150,
    show: bool = True,
    json_output: str | None = None,
) -> dict[str, Any] | None:
    """Analyze training checkpoints and generate visualizations.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        output_path: Path for output image.
        smoothing_window: Window size for loss smoothing.
        figsize: Figure size in inches.
        dpi: Figure resolution.
        show: Whether to display the plot.
        json_output: Optional path for JSON metrics export.

    Returns:
        Summary dictionary if successful, None otherwise.
    """
    checkpoints = find_checkpoints(checkpoint_dir)

    if not checkpoints:
        logger.error(f"No checkpoints found in {checkpoint_dir}")
        return None

    logger.info(f"Found {len(checkpoints)} checkpoint(s)")

    # Use the latest checkpoint
    latest = checkpoints[-1]
    logger.info(f"Analyzing: {latest}")

    state = load_trainer_state(latest)
    if state is None:
        return None

    log_history = state.get("log_history", [])
    if not log_history:
        logger.error("No log history found in trainer state")
        return None

    metrics = extract_metrics(log_history)

    if not metrics["losses"]:
        logger.error("No loss values found in log history")
        return None

    # Compute and print summary
    summary = compute_summary(state, metrics)
    print_summary(summary)

    # Create visualization
    create_visualization(
        metrics,
        output_path,
        smoothing_window=smoothing_window,
        figsize=figsize,
        dpi=dpi,
        show=show,
    )

    # Export to JSON if requested
    if json_output:
        export_data = {
            "summary": summary,
            "metrics": metrics,
        }
        with open(json_output, "w") as f:
            json.dump(export_data, f, indent=2)
        logger.info(f"Exported metrics to: {json_output}")

    return summary


def main() -> None:
    """Main entry point for the analysis script."""
    args = parse_args()

    # Parse figsize
    try:
        width, height = map(float, args.figsize.split(","))
        figsize = (width, height)
    except ValueError:
        logger.warning(f"Invalid figsize '{args.figsize}', using default (14, 10)")
        figsize = (14, 10)

    result = analyze_training(
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        smoothing_window=args.smoothing_window,
        figsize=figsize,
        dpi=args.dpi,
        show=not args.no_display,
        json_output=args.json_output,
    )

    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
