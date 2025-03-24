import os
import argparse
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from ..utils.dataloader import ImageList
from ..utils.preprocess import val_transform

plt.rcParams["font.family"] = "serif"

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Confusion Matrix for a Classification Model"
    )
    parser.add_argument(
        "--val_dir", type=str, default="../data/val",
        help="Path to the validation directory"
    )
    parser.add_argument(
        "--model", type=str,
        default="timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288",
        help="Model name from timm"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="./HW1/checkpoints_report/cp_bz_lr/"
                "cp_seresnextaa101d_bz16_lr0.00001_sh-expo/best_model.pth",
        help="Path to the model checkpoint (.pth)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_classes", type=int, default=100,
        help="Number of classes"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = timm.create_model(
        args.model, pretrained=False, num_classes=args.num_classes
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Prepare validation dataset
    transform = val_transform(334, 320)  # Use dimensions consistent with training
    val_dataset = ImageList(args.val_dir, transform_w=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["img_w"].to(device)
            labels = batch["target"].to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    print(classification_report(all_labels, all_preds))

    # Plot confusion matrix with Seaborn heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        cm,
        annot=False,  # Turn off numeric annotations in each cell
        fmt="d",
        cmap="Blues",
        xticklabels=np.arange(args.num_classes),
        yticklabels=np.arange(args.num_classes)
    )

    # Set labels and title with larger font sizes
    ax.set_xlabel("Predicted Label", fontsize=16)
    ax.set_ylabel("True Label", fontsize=16)
    ax.set_title("Confusion Matrix of the Validation Set", fontsize=18)

    # Increase the x/y tick label font sizes
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

    # Capture the color bar object to increase its font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    # (Optional) Add a label to the color bar
    # cbar.set_label("Number of Samples", fontsize=16)

    plt.tight_layout()
    plt.savefig("./HW1/visualize/confusion_matrix.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
