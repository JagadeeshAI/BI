import matplotlib.pyplot as plt

# Dummy data: use your actual accuracy numbers and steps
bars = [
    {'step': 0, 'start': 0,  'end': 49, 'acc': 59.76, 'solid': True, 'color': 'tab:blue'},    # Pretrain (0-49)
    {'step': 1, 'start': 10, 'end': 59, 'acc': 56.19, 'solid': True, 'color': 'orange'},      # Step 1 (10-59, orange)
    {'step': 2, 'start': 20, 'end': 69, 'acc': None, 'solid': False, 'color': 'gray'},        # Step 2
    {'step': 3, 'start': 30, 'end': 79, 'acc': None, 'solid': False, 'color': 'gray'},        # Step 3
    {'step': 4, 'start': 40, 'end': 89, 'acc': None, 'solid': False, 'color': 'gray'},        # Step 4
    {'step': 5, 'start': 50, 'end': 99, 'acc': None, 'solid': False, 'color': 'gray'},        # Step 5
]

fig, ax = plt.subplots(figsize=(12, 5))

for bar in bars:
    y = bar['step']
    color = bar['color']
    linestyle = '-' if bar['solid'] else 'dotted'
    alpha = 0.7 if bar['solid'] else 0.3
    ax.barh(
        y=y, width=bar['end'] - bar['start'] + 1, left=bar['start'],
        height=0.8, color=color, edgecolor='black', alpha=alpha, linestyle=linestyle
    )
    # Annotate with accuracy, if available
    if bar['acc'] is not None:
        ax.text(
            (bar['start'] + bar['end']) / 2, y,
            f"{bar['acc']}%", va='center', ha='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.7, alpha=0.8)
        )

# Set axes, labels, grid
ax.set_yticks([b['step'] for b in bars])
ax.set_yticklabels([f"Step {b['step']}" for b in bars])
ax.set_xlabel("Class Index")
ax.set_ylabel("Sliding Window Step")
ax.set_xlim(-5, 105)
ax.set_ylim(-1, len(bars))
ax.set_title("Sliding Window: Sequential Class Replacement and Accuracy")
ax.grid(axis='x', linestyle=':', alpha=0.7)

plt.tight_layout()

# Save the figure
plt.savefig('sliding_window_accuracy.png', dpi=300)
plt.close()  # Closes the plot so it doesn't show as a pop-up
