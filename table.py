import pandas as pd
import matplotlib.pyplot as plt

# Copy your data into Python (from your JSON)
data = [
  {"step": "Pretrained", "retain_classes": "0-49", "unlearn_classes": None, "LA": 0.5636623748211731, "UA": None, "FM": None, "KL-D": 1.0835573799449827, "MIA": None, "RTE": None},
  {"step": "Step1", "retain_classes": "10-59", "unlearn_classes": "0-9", "LA": 0.5077639751552795, "UA": 0.0, "FM": -0.055898399665893606, "KL-D": 1.551244130045731, "MIA": 0.07459751238615757, "RTE": None},
  {"step": "Step2", "retain_classes": "20-69", "unlearn_classes": "10-19", "LA": 0.4672897196261682, "UA": 0.0, "FM": -0.04047425552911127, "KL-D": 1.9225691285831534, "MIA": 0.07409864284907101, "RTE": None},
  {"step": "Step3", "retain_classes": "30-79", "unlearn_classes": "20-29", "LA": 0.4623493975903614, "UA": 0.0, "FM": -0.004940322035806799, "KL-D": 1.844005814517837, "MIA": 0.09104278253645562, "RTE": None},
  {"step": "Step4", "retain_classes": "40-89", "unlearn_classes": "30-39", "LA": 0.4780701754385965, "UA": 0.0, "FM": 0.015720777848235057, "KL-D": 1.7897456542790284, "MIA": 0.08140485984503114, "RTE": None},
  {"step": "Step5", "retain_classes": "50-99", "unlearn_classes": "40-49", "LA": 0.48345323741007196, "UA": 0.0, "FM": 0.005383061971475478, "KL-D": 1.5544607478080037, "MIA": 0.03238606566204114, "RTE": None}
]

df = pd.DataFrame(data)

# Print as pretty table
print(df[["step", "retain_classes", "unlearn_classes", "LA", "UA", "FM", "KL-D", "MIA"]].to_markdown(index=False))

# Optional: Save as CSV for sharing/analysis
df.to_csv("results_table.csv", index=False)
