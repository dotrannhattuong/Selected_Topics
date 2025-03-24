import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    ExponentialLR, MultiStepLR, CosineAnnealingLR
)
from torch.utils.data import DataLoader

import timm
from tqdm import tqdm
from termcolor import colored

from log_utils.utils import ReDirectSTD
from utils.dataloader import ImageList, ImageList_test
from utils.preprocess import val_transform
from utils.utils import print_model_size, validate, test
from utils.sampler import sampler_imbl


def main():
    """Main function to train, validate, and test the classification model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Classification")
    # Data parameters
    parser.add_argument(
        "--train_dir", type=str, default="./data/train",
        help="Path to the training directory"
    )
    parser.add_argument(
        "--test_dir", type=str, default="./data/test",
        help="Path to the test directory"
    )
    parser.add_argument(
        "--val_dir", type=str, default="./data/val",
        help="Path to the validation directory"
    )
    parser.add_argument(
        "--resize_size", type=int, default=334,
        help="Resize dimension for images"
    )
    parser.add_argument(
        "--crop_size", type=int, default=320,
        help="Crop size for images"
    )
    # Model parameters
    parser.add_argument(
        "--model", type=str,
        default="timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288",
        help="Model name from timm"
    )
    # Training parameters
    parser.add_argument(
        "--device", default="cuda:2",
        help="Device to run training on"
    )
    parser.add_argument(
        "--bz", type=int, default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=300,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--schedule", type=str, default="expo",
        choices=["expo", "multi", "cosine"],
        help="Learning rate scheduler type"
    )
    # Save directory parameters
    parser.add_argument(
        "--save_dir", type=str, default="cp_seresnextaa101d",
        help="Directory to save checkpoints and logs"
    )

    args = parser.parse_args()

    # Set device for training
    device = torch.device(args.device)
    print(colored(
        f"Loading model: {args.model}",
        color="red", force_color=True
    ))

    # Prepare save directory and logging
    save_dir = f"{args.save_dir}_sh-{args.schedule}_imbl"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")
    log_dir = os.path.join(save_dir, "logs.txt")
    ReDirectSTD(log_dir, "stdout", True)

    # Load model from timm with pretrained weights
    model = timm.create_model(
        args.model, pretrained=True, num_classes=100
    )
    data_config = timm.data.resolve_model_data_config(model)
    transforms_train = timm.data.create_transform(
        **data_config, is_training=True
    )

    print_model_size(model)
    model.to(device)
    model.train()

    # Load datasets using custom dataloader classes
    print(colored(
        f"Loading datasets from {args.train_dir}, {args.val_dir}, "
        f"{args.test_dir}",
        color="blue", force_color=True
    ))
    resize_size = args.resize_size
    crop_size = args.crop_size
    train_dataset = ImageList(
        args.train_dir, transform_w=transforms_train
    )
    val_dataset = ImageList(
        args.val_dir, transform_w=val_transform(resize_size, crop_size)
    )
    test_dataset = ImageList_test(
        args.test_dir, transform=val_transform(resize_size, crop_size)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.bz,
        sampler=sampler_imbl(train_dataset),
        num_workers=4, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.bz,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.bz,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Choose learning rate scheduler based on the provided argument
    if args.schedule == "expo":
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif args.schedule == "multi":
        scheduler = MultiStepLR(
            optimizer, milestones=[30, 80], gamma=0.1
        )
    elif args.schedule == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    else:
        raise ValueError("Invalid scheduler type")

    best_valid_acc = 0.0
    epochs = args.epochs
    max_iters = len(train_loader)

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        for step, batch_train in enumerate(train_loader):
            # Move training data to device
            train_w = batch_train["img_w"].to(device)
            train_labels = batch_train["target"].to(device)
            lr = optimizer.param_groups[0]["lr"]

            # Forward pass
            outputs = model(train_w)
            loss = criterion(outputs, train_labels)

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training status every 20 steps or at the last iteration
            if step % 20 == 0 or step == max_iters - 1:
                print(
                    "Epoch {} Iters: ({}/{}) \t Loss Sup = {:<10.6f} \t "
                    "learning rate = {:<10.6f}".format(
                        epoch, step, max_iters, loss.item(), lr
                    )
                )

        # Step the learning rate scheduler at the end of the epoch
        scheduler.step()

        # Validation phase
        print(f"Validating at epoch {epoch}")
        val_acc = validate(model, val_loader, device)
        print(f"Validation accuracy: {val_acc * 100:.3f}%")

        # Save model if validation accuracy improves
        if val_acc >= best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val accuracy: {best_valid_acc:.5f}")

            # Test the model and save predictions
            print("Testing...")
            test(
                model, test_loader, device,
                output_csv=os.path.join(save_dir, "prediction.csv")
            )

        # Log best validation accuracy
        log_str = (
            "=====================================\n"
            f"Best Validation Accuracy: {best_valid_acc * 100:.3f}%\n"
            "====================================="
        )
        print(colored(log_str, color="red", force_color=True))


if __name__ == "__main__":
    main()
