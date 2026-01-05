#!/usr/bin/env python3
import json
import os
import sys
import matplotlib.pyplot as plt

def analyze_training(checkpoint_dir):
    checkpoints = []
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            if item.startswith("checkpoint-"):
                checkpoints.append(os.path.join(checkpoint_dir, item))
    
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoint(s)")
    
    latest = checkpoints[-1]
    state_file = os.path.join(latest, "trainer_state.json")
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    log_history = state.get("log_history", [])
    
    steps, losses, learning_rates, grad_norms = [], [], [], []
    
    for entry in log_history:
        if "loss" in entry:
            steps.append(entry.get("step", 0))
            losses.append(entry["loss"])
            learning_rates.append(entry.get("learning_rate", 0))
            grad_norms.append(entry.get("grad_norm", 0))
    
    print("\n" + "=" * 60)
    print("MiniCrit-7B Training Analysis")
    print("Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3")
    print("=" * 60)
    
    print(f"\n📊 TRAINING SUMMARY")
    print(f"   Steps completed: {state.get('global_step', 0):,}")
    print(f"   Steps planned: {state.get('max_steps', 0):,}")
    print(f"   Progress: {state.get('global_step', 0) / state.get('max_steps', 1) * 100:.1f}%")
    
    print(f"\n📈 LOSS METRICS")
    print(f"   Initial loss: {losses[0]:.4f}")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   Reduction: {(1 - losses[-1]/losses[0]) * 100:.1f}%")
    
    print(f"\n🎯 GRADIENT METRICS")
    valid_grads = [g for g in grad_norms if g > 0]
    print(f"   Avg grad norm: {sum(valid_grads)/len(valid_grads):.4f}")
    
    print(f"\n🖥️  Hardware: NVIDIA H100 (Lambda Labs GPU Grant)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MiniCrit-7B Training Analysis\nAntagon Inc. | Trained with Lambda Labs GPU Grant', fontsize=14, fontweight='bold')
    
    # Loss curve
    ax1 = axes[0, 0]
    ax1.plot(steps, losses, 'b-', linewidth=0.8, alpha=0.7)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(losses) * 1.1)
    
    # Loss curve (smoothed)
    ax2 = axes[0, 1]
    window = 20
    smoothed = []
    for i in range(len(losses)):
        start = max(0, i - window // 2)
        end = min(len(losses), i + window // 2)
        smoothed.append(sum(losses[start:end]) / (end - start))
    ax2.plot(steps, smoothed, 'b-', linewidth=1.5)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss (Smoothed)')
    ax2.set_title('Training Loss (Smoothed, window=20)')
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    ax3 = axes[1, 0]
    ax3.plot(steps, learning_rates, 'g-', linewidth=0.8)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule (Cosine)')
    ax3.grid(True, alpha=0.3)
    
    # Gradient norm
    ax4 = axes[1, 1]
    ax4.plot(steps, grad_norms, 'r-', linewidth=0.8, alpha=0.7)
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Gradient Norm')
    ax4.set_title('Gradient Norm Over Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved: training_curves.png")
    plt.show()

if __name__ == "__main__":
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "minicrit_7b_output"
    analyze_training(checkpoint_dir)
