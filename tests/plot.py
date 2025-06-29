import os
import json
import matplotlib.pyplot as plt

# Configuration
results_dir = "results/steps"
oracle_dir = "results/oracle"
output_plot = "sliding_window_accuracy.png"

bars = []

# Steps 1–5
for step in range(1, 6):
    step_start = 10 * step
    step_end = 49 + 10 * step

    # Load our accuracy
    ours_path = os.path.join(results_dir, f"step{step}.json")
    if not os.path.exists(ours_path):
        print(f"⚠️ Missing: {ours_path}")
        acc = None
    else:
        with open(ours_path, "r") as f:
            acc = json.load(f).get("overall_acc", None)

    # Load oracle accuracy
    oracle_path = os.path.join(oracle_dir, f"{step_start}_{step_end}.json")
    oracle_acc = None
    if os.path.exists(oracle_path):
        with open(oracle_path, "r") as f:
            oracle_acc = json.load(f).get("overall_acc", None)

    bars.append({
        'step': step,
        'start': step_start,
        'end': step_end,
        'acc': acc,
        'oracle': oracle_acc,
        'solid': acc is not None,
        'color': 'tab:orange' if acc is not None else 'gray'
    })

# Step 0 manually
bars.insert(0, {
    'step': 0,
    'start': 0,
    'end': 49,
    'acc': 59.76,
    'oracle': 65.95,
    'solid': True,
    'color': 'tab:orange'
})

# Plotting
fig, ax = plt.subplots(figsize=(12, 5))

for bar in bars:
    y = bar['step']
    alpha = 0.8 if bar['solid'] else 0.3
    width = bar['end'] - bar['start'] + 1
    center = (bar['start'] + bar['end']) / 2

    ax.barh(
        y=y,
        width=width,
        left=bar['start'],
        height=0.8,
        color=bar['color'],
        edgecolor='black',
        alpha=alpha
    )

    # Combined accuracy text (ours/oracle)
    if bar['acc'] is not None and bar['oracle'] is not None:
        acc_text = f"{bar['acc']:.2f} / {bar['oracle']:.2f} %"
        ax.text(
            center, y,
            acc_text,
            va='center', ha='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.7, alpha=0.85)
        )

# Axes and layout
ax.set_yticks([b['step'] for b in bars])
ax.set_yticklabels([f"Step {b['step']}" for b in bars])
ax.set_xlabel("Class Index")
ax.set_ylabel("Sliding Window Step")
ax.set_xlim(-5, 110)
ax.set_ylim(-1, len(bars))
ax.set_title("Sliding Window: Class Retention Accuracy (Ours / Oracle)")
ax.grid(axis='x', linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.close()

print(f"✅ Plot saved to {output_plot}")
