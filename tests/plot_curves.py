# tests/plot_curves.py

import glob, json
import pandas as pd
import matplotlib.pyplot as plt

def load_hist(path):
    with open(path) as f:
        hist = json.load(f)
    # assume hist = { "train_loss": [...], "val_acc": [...], ... }
    df = pd.DataFrame(hist)
    df['split'] = path.split('_')[-1].replace('.json','')
    return df

all_hist = pd.concat([load_hist(p) for p in glob.glob("results/oracle/*.json")], ignore_index=True)
for split, grp in all_hist.groupby('split'):
    plt.plot(grp['epoch'], grp['val_acc'], label=f"{split} val")
plt.xlabel("Epoch")
plt.ylabel("Val Accuracy")
plt.legend()
plt.title("Oracle Val Accuracy by Split")
plt.show()
