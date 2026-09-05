from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CrossValidationDataset(Dataset):
    def __init__(self, metadata, processed_dir, transform=None):
        self.metadata = metadata.reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.transform = transform

        self.classes = sorted(self.metadata["dx"].unique())
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        self.samples = []

        for _, row in self.metadata.iterrows():
            path = (
                self.processed_dir
                / row["source_split"]
                / row["dx"]
                / f"{row['image_id']}_224x224.jpg"
            )

            label_index = self.class_to_idx[row["dx"]]

            self.samples.append((str(path), label_index))

        self.targets = [
            label_index
            for _, label_index in self.samples
        ]

        self.targets = [
            self.class_to_idx[dx]
            for dx in self.metadata["dx"]
        ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path, label_index = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label_index