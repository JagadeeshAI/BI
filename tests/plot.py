import os
import json
import matplotlib.pyplot as plt

# Configuration
results_dir = "results/steps"
oracle_dir = "results/oracle"
output_plot = "EX1.png"

bars = []

# Pretrained baseline (step 0)
pretrained_path = os.path.join(oracle_dir, "0_49.json")
if os.path.exists(pretrained_path):
    with open(pretrained_path, "r") as f:
        data = json.load(f)
        pretrained_acc = data.get("overall_acc", None)

        bars.append({
            'step': 0,
            'start': 0,
            'end': 49,
            'acc': pretrained_acc,
            'oracle': None,
            'solid': True,
            'color': 'tab:blue'
        })
else:
    print(f"⚠️ Missing: {pretrained_path}")

# Sliding window steps 1–5
for step in range(1, 6):
    start_class = step * 10
    end_class = start_class + 49

    ours_path = os.path.join(results_dir, f"step{step}.json")
    acc = None
    if os.path.exists(ours_path):
        with open(ours_path, "r") as f:
            ours_data = json.load(f)
            acc = ours_data.get("overall_acc", None)
    else:
        print(f"⚠️ Missing: {ours_path}")

    # Get oracle accuracy using corrected range
    oracle_path = os.path.join(oracle_dir, f"{start_class}_{end_class}.json")
    oracle_acc = None
    if os.path.exists(oracle_path):
        with open(oracle_path, "r") as f:
            oracle_data = json.load(f)
            oracle_acc = oracle_data.get("overall_acc", None)

    bars.append({
        'step': step,
        'start': start_class,
        'end': end_class,
        'acc': acc,
        'oracle': oracle_acc,
        'solid': acc is not None,
        'color': 'tab:orange' if acc is not None else 'gray'
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
        height=0.7,
        color=bar['color'],
        edgecolor='black',
        alpha=alpha
    )

    # Accuracy text
    if bar['step'] == 0 and bar['acc'] is not None:
        acc_text = f"{bar['acc']:.2f} %"
    elif bar['acc'] is not None and bar['oracle'] is not None:
        acc_text = f"{bar['acc']:.2f} / {bar['oracle']:.2f} %"
    else:
        acc_text = None

    if acc_text:
        ax.text(
            center, y,
            acc_text,
            va='center', ha='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.7, alpha=0.85)
        )

# Axes and layout
ax.set_yticks([b['step'] for b in bars])
ax.set_yticklabels([
    "Pretrained" if b['step'] == 0 else f"Step {b['step']}"
    for b in bars
])
ax.set_xlabel("Class Index")
ax.set_ylabel("Sliding Window Step")

# X-ticks: 0, 10, ..., 100
x_ticks = list(range(0, 110, 10))
ax.set_xticks(x_ticks)
ax.set_xticklabels([str(x - 1) if x != 0 else "0" for x in x_ticks])
ax.set_xlim(-5, 105)
ax.set_ylim(-1, len(bars) + 1)
ax.set_title("Sliding Window: Class Retention Accuracy (Ours / Oracle)")
ax.grid(axis='x', linestyle=':', alpha=0.7)


plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(output_plot, dpi=300)
plt.close()

print(f"✅ Plot saved to {output_plot}")
