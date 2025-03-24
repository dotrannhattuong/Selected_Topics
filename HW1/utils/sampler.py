from collections import Counter
import torch
from torch.utils.data import WeightedRandomSampler


def sampler_imbl(train_dataset):
    # Retrieve labels from the training dataset
    labels = train_dataset.labels

    # Count the frequency of each label using Counter
    counts = Counter(labels)

    # Calculate class weights: if a class does not appear, assign a weight of
    # 0.0, otherwise assign the inverse of its count.
    class_weights = {}
    for c in range(100):
        if counts[c] == 0:
            class_weights[c] = 0.0
        else:
            class_weights[c] = 1.0 / counts[c]

    # Generate a weight for each sample based on its corresponding class weight
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    # Create the WeightedRandomSampler using the computed sample weights.
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler
