from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


def build_image_dataset(data_dir, transform=None):
    data_dir = Path(data_dir)
    dataset = ImageFolder(data_dir, transform=transform)
    return dataset


def inspect_dataset_item(dataset, index):
    image, label_index = dataset[index]
    image_type = str(type(image))
    class_name = dataset.classes[label_index]
    result = {
        "image_type": image_type,
        "label_index": label_index,
        "class_name": class_name
    }
    return result


def build_data_loader(dataset, batch_size, shuffle):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def build_split_datasets(processed_dir, transform):
    processed_dir = Path(processed_dir)
    train_dataset = build_image_dataset(processed_dir / "train", transform=transform)
    val_dataset = build_image_dataset(processed_dir / "val", transform=transform)
    test_dataset = build_image_dataset(processed_dir / "test", transform=transform)
    result = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }
    return result


def build_split_loaders(datasets, batch_size):
    train_loader = build_data_loader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = build_data_loader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = build_data_loader(datasets["test"], batch_size=batch_size, shuffle=False)

    result = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    return result
    

if __name__ == "__main__":
    tensor_transform = transforms.ToTensor()

    train_dataset = build_image_dataset("data/processed/train", transform=tensor_transform)

    image, label_index = train_dataset[0]

    print(type(image))
    print(image.shape)
    print(label_index)
    train_loader = build_data_loader(train_dataset, batch_size=8, shuffle=True)

    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)
    datasets = build_split_datasets("data/processed", transform=tensor_transform)

    print(len(datasets["train"]))
    print(len(datasets["val"]))
    print(len(datasets["test"]))
    loaders = build_split_loaders(datasets, batch_size=8)

    train_images, train_labels = next(iter(loaders["train"]))
    val_images, val_labels = next(iter(loaders["val"]))
    test_images, test_labels = next(iter(loaders["test"]))

    print(train_images.shape, train_labels.shape)
    print(val_images.shape, val_labels.shape)
    print(test_images.shape, test_labels.shape)