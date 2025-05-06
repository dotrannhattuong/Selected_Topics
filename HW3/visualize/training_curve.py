import json
import matplotlib.pyplot as plt

# Path to training log JSON
json_file = "metrics.json"

# Containers for iteration, loss, and AP
iterations_loss = []
losses = []

iterations_ap = []
ap_values = []

# Parse the file line by line
with open(json_file, "r") as f:
    for line in f:
        try:
            entry = json.loads(line)
            it = entry.get("iteration", None)

            # Collect total_loss data
            if it is not None and "total_loss" in entry and it <= 50000:
                iterations_loss.append(it)
                losses.append(entry["total_loss"])

            # Collect segm/AP from validation results
            if it is not None and "segm/AP" in entry and it <= 50000:
                iterations_ap.append(it)
                ap_values.append(entry["segm/AP"])

        except json.JSONDecodeError:
            continue

from matplotlib.ticker import FuncFormatter

# === Plot 1: Training Loss ===
plt.figure(figsize=(10, 8))
plt.plot(iterations_loss, losses, label="Total Loss", color='tab:red')
plt.xlabel("Iteration", fontsize=30)
plt.ylabel("Total Loss", fontsize=30)
plt.title("Training Loss Curve (up to Iteration 50,000)", fontsize=16)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000)}K'))  # Format x-axis
plt.grid(True)
plt.legend(fontsize=28)
plt.tight_layout()
plt.savefig("training_curve_loss.pdf", format="pdf")

# === Plot 2: segm/AP ===
plt.figure(figsize=(10, 8))
plt.plot(iterations_ap, ap_values, label="segm/AP", color='tab:blue', marker='o', linestyle='--')
plt.xlabel("Iteration", fontsize=30)
plt.ylabel("segm/AP", fontsize=30)
plt.title("Segmentation AP (Validation) (up to Iteration 50,000)", fontsize=16)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000)}K'))  # Format x-axis
plt.grid(True)
plt.legend(fontsize=28)
plt.tight_layout()
plt.savefig("training_curve_ap.pdf", format="pdf")

print("Saved: training_curve_loss.pdf and training_curve_ap.pdf")
