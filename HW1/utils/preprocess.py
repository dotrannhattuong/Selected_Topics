from torchvision import transforms


def strong_transform(crop_size=224):
    return transforms.Compose([
        transforms.RandAugment(3, 9),
        transforms.RandomResizedCrop(
            size=(crop_size, crop_size),
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def train_transform(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def val_transform(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
