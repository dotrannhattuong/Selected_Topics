import os
import re
import argparse
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"


def strip_ansi_codes(line):
    """
    Remove ANSI escape sequences (color codes) from a given line of text.
    """
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
    return ansi_escape.sub('', line)


def parse_logs_accuracy(log_file):
    """
    Parse the log file to get validation accuracy per epoch.

    Assumes lines like:
      Validating at epoch 0
      Validation accuracy: 12.345
    """
    validating_pattern = re.compile(r"Validating at epoch\s+(\d+)")
    accuracy_pattern = re.compile(r"Validation accuracy:\s*([\d\.]+)")

    epoch_acc = {}
    current_epoch = None

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = strip_ansi_codes(line)
            vm = validating_pattern.search(clean_line)
            if vm:
                current_epoch = int(vm.group(1))
            else:
                am = accuracy_pattern.search(clean_line)
                if am and current_epoch is not None:
                    acc_value = float(am.group(1))
                    epoch_acc[current_epoch] = acc_value
                    current_epoch = None

    epochs = sorted(epoch_acc.keys())
    accuracies = [epoch_acc[e] for e in epochs]
    return epochs, accuracies


def parse_logs_epoch(log_file):
    """
    Parse the log file to compute average loss per epoch.

    Assumes lines like:
    Epoch 0 Iters: (20/1295)  Loss Sup = 4.559049  learning rate = 0.000010
    """
    pattern = re.compile(
        r"Epoch\s+(\d+)\s+Iters:\s*\((\d+)/(\d+)\)\s+Loss Sup = ([\d\.]+)\s+"
        r"learning rate = ([\d\.]+)"
    )
    epoch_losses = {}

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = strip_ansi_codes(line)
            match = pattern.search(clean_line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(4))
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = []
                epoch_losses[epoch].append(loss)

    epochs = sorted(epoch_losses.keys())
    avg_losses = []
    for e in epochs:
        avg_loss = sum(epoch_losses[e]) / len(epoch_losses[e])
        avg_losses.append(avg_loss)

    return epochs, avg_losses


def plot_accuracy_curve_epoch(epochs, accuracies, output):
    """
    Plot and save the epoch-wise validation accuracy curve,
    showing only thick spines on x and y axes.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(epochs, accuracies, marker='o', linestyle='-', linewidth=3)

    ax.set_xlabel("Epoch", fontsize=24)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=24)
    ax.set_title("Validation Accuracy Curve", fontsize=26)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(True)

    # Hide top and right spines, make bottom/left spines thicker
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curve_epoch(epochs, avg_losses, output):
    """
    Plot and save the epoch-wise training curve (epoch vs. average loss),
    showing only thick spines on x and y axes.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(epochs, avg_losses, marker='o', linestyle='-', linewidth=3)

    ax.set_xlabel("Epoch", fontsize=24)
    ax.set_ylabel("Average Loss", fontsize=24)
    ax.set_title("Training Curve", fontsize=26)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(True)

    # Hide top and right spines, make bottom/left spines thicker
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Loss & Accuracy Curves"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=(
            "/mnt/HDD1/tuong/selected/Selected_Topics/HW1/"
            "checkpoints_report/cp_bz_lr/"
            "cp_seresnextaa101d_bz16_lr0.00001_sh-expo/logs.txt"
        ),
        help="Path to the training log file (logs.txt)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="HW1/visualize",
        help="Directory to save the resulting plots."
    )
    args = parser.parse_args()

    # Parse logs for accuracy and loss
    epochs_acc, accuracies = parse_logs_accuracy(args.log_file)
    epochs_loss, avg_losses = parse_logs_epoch(args.log_file)

    # Filter for only the first 100 epochs
    epochs_acc_100, acc_100 = [], []
    for e, acc_val in zip(epochs_acc, accuracies):
        if e < 100:
            epochs_acc_100.append(e)
            acc_100.append(acc_val)

    epochs_loss_100, avg_losses_100 = [], []
    for e, loss_val in zip(epochs_loss, avg_losses):
        if e < 100:
            epochs_loss_100.append(e)
            avg_losses_100.append(loss_val)

    # Plot the first 100 epochs for accuracy
    accuracy_out = os.path.join(args.output_dir, "accuracy_curve.pdf")
    plot_accuracy_curve_epoch(epochs_acc_100, acc_100, output=accuracy_out)

    # Plot the first 100 epochs for loss
    loss_out = os.path.join(args.output_dir, "training_curve.pdf")
    plot_training_curve_epoch(
        epochs_loss_100,
        avg_losses_100,
        output=loss_out
    )


if __name__ == "__main__":
    main()
