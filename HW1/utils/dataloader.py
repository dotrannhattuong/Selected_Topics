import os
from PIL import Image
from torch.utils.data import Dataset


class ImageList(Dataset):
    """Dataset for loading images and corresponding labels from a directory.

    The root directory should contain subdirectories with names
    representing labels. Only images with extensions .jpg, .jpeg,
    .png, .bmp will be processed.
    """

    def __init__(self, root_dir, transform_w=None, transform_str=None):
        """
        Initialize the ImageList dataset.

        Args:
            root_dir (str): Root directory with subdirectories for each label.
            transform_w (callable, optional): Weak transformation function.
            transform_str (callable, optional): Strong transformation function.
        """
        self.root_dir = root_dir
        self.transform_w = transform_w
        self.transform_str = transform_str
        self.image_paths = []
        self.labels = []

        # Iterate over sorted label directories
        for label_str in sorted(os.listdir(self.root_dir)):
            label_dir = os.path.join(self.root_dir, label_str)
            if os.path.isdir(label_dir):
                label = int(label_str)
                for file_name in os.listdir(label_dir):
                    # Process only image files with supported extensions
                    if file_name.lower().endswith(
                        ('.jpg', '.jpeg', '.png', '.bmp')
                    ):
                        file_path = os.path.join(label_dir, file_name)
                        self.image_paths.append(file_path)
                        self.labels.append(label)

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            dict: Dictionary with keys 'img_w', 'img_str' (if available),
            'target', and 'image_name'.
        """
        output = {}
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply wide transformation if available; otherwise use the image.
        if self.transform_w:
            image_w = self.transform_w(image)
        else:
            image_w = image

        # Apply strong transformation if available.
        if self.transform_str:
            image_str = self.transform_str(image)
            output["img_str"] = image_str

        output["img_w"] = image_w
        output["target"] = label
        output["image_name"] = os.path.basename(img_path)
        return output


class ImageList_test(Dataset):
    """Dataset for loading test images from a single directory.

    Only images with extensions .jpg, .jpeg, .png, .bmp will be processed.
    """

    def __init__(self, test_dir, transform=None):
        """
        Initialize the ImageList_test dataset.

        Args:
            test_dir (str): Directory containing test images.
            transform (callable, optional): Transformation function to apply.
        """
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = []

        # Iterate over sorted image files in the test directory.
        for file_name in sorted(os.listdir(self.test_dir)):
            if file_name.lower().endswith(
                ('.jpg', '.jpeg', '.png', '.bmp')
            ):
                full_path = os.path.join(self.test_dir, file_name)
                self.image_paths.append(full_path)

        # Ensure the image paths are sorted.
        self.image_paths.sort()

    def __len__(self):
        """Return the total number of test images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieve a test image and its metadata by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            dict: Dictionary with keys 'img' and 'image_name'.
        """
        output = {}
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        image_name = os.path.splitext(
            os.path.basename(img_path)
        )[0]

        output["img"] = image
        output["image_name"] = image_name
        return output
