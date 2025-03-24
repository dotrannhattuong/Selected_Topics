import torch
from tqdm import tqdm
import pandas as pd
import zipfile
from termcolor import colored


def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    # Print model size in millions using green colored text.
    print(colored(
        f"Total parameters: {total_params/1e6:.2f} M",
        color="green",
        force_color=True
    ))
    # Ensure model size does not exceed 100M parameters.
    assert total_params <= 100e6, (
        "Model size exceeds 100M parameters! Current size: " +
        f"{total_params/1e6:.2f} M"
    )


def validate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    # Disable gradient computation for validation.
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating"):
            # Move images and labels to the specified device.
            images = batch_data["img_w"].to(device)
            labels = batch_data["target"].to(device)
            outputs = model(images)

            # Obtain predictions by selecting the index with the maximum logit.
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate predictions and labels from all batches.
    labels_tensor = torch.cat(all_labels, dim=0)
    predicts_tensor = torch.cat(all_preds, dim=0)

    # Calculate accuracy.
    accuracy = (
        torch.sum(predicts_tensor == labels_tensor).item() /
        len(labels_tensor)
    )
    model.train()
    return accuracy


def test(model, dataloader, device, output_csv):
    model.eval()
    predictions = []

    # Disable gradient computation during testing.
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Testing"):
            images = batch_data["img"].to(device)
            outputs = model(images)

            # Get the predicted class indices.
            _, preds = torch.max(outputs, 1)

            # Collect predictions with their corresponding image names.
            image_names = batch_data["image_name"]
            for img_name, pred in zip(image_names, preds):
                predictions.append({
                    "image_name": img_name,
                    "pred_label": int(pred)
                })

    # Save predictions to a CSV file.
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

    # Create a zip archive containing the CSV file.
    folder_name = '/'.join(output_csv.split('/')[:-1])
    basename = folder_name.split('/')[-1]
    zip_filename = (
        f"{folder_name}/{basename}.zip"
    )
    with zipfile.ZipFile(
        zip_filename, 'w', zipfile.ZIP_DEFLATED
    ) as zipf:
        zipf.write(output_csv, arcname="prediction.csv")

    print(
        f"Saved zipped predictions to {zip_filename}"
    )
    model.train()
