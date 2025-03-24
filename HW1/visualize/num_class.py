import os
import numpy as np
import matplotlib.pyplot as plt

# Set the font to Times New Roman and increase the font size globally.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16


def count_images_per_class(folder):
    """
    Count the number of images in each class within a folder.

    Assumes each subfolder of the given folder represents a class, with class
    names as numbers (e.g., '0', '1', ..., '99').

    Args:
        folder (str): Path to the folder containing subfolders for each class.

    Returns:
        dict: A dictionary where keys are class names and values are the number
              of images in that class.
    """
    counts = {}
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            image_files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            counts[class_name] = len(image_files)
    return counts


def plot_class_distribution(counts, title, pdf_filename):
    """
    Plot a bar chart showing the number of images per class.

    Args:
        counts (dict): Dictionary with class names as keys and image counts
                       as values.
        title (str): Title of the bar chart.
        pdf_filename (str): Filename to save the plot as a PDF.
    """
    # Sort class names numerically (assuming they are strings of digits).
    classes = sorted(counts.keys(), key=lambda x: int(x))
    image_counts = [counts[cls] for cls in classes]

    x = np.arange(len(classes))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(x, image_counts, width, color='mediumseagreen')
    ax.set_xlim(0, len(classes))
    ax.set_xlabel('Class ID', fontsize=20)
    ax.set_ylabel('Number of Images', fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # Fade the outer box by reducing alpha on spines
    for spine in ax.spines.values():
        spine.set_alpha(0.3)

    plt.tight_layout()
    plt.savefig(pdf_filename, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Define paths to the original and augmented training data folders.
    train_folder = "../data/train"
    train_aug_folder = "../data/train_aug"

    # Count images per class for both folders.
    counts_train = count_images_per_class(train_folder)
    counts_train_aug = count_images_per_class(train_aug_folder)

    # Plot and save the class distribution charts.
    plot_class_distribution(
        counts_train,
        "Class Distribution Before Augmentation",
        "HW1/visualize/class_distribution_before.pdf"
    )
    plot_class_distribution(
        counts_train_aug,
        "Class Distribution After Augmentation",
        "HW1/visualize/class_distribution_after.pdf"
    )
