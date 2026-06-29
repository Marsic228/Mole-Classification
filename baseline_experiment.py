from datetime import datetime
from pathlib import Path
import json

from torchvision import transforms

from dataset_loader_check import build_split_datasets, build_split_loaders
from baseline_cnn_forward_check import build_baseline_cnn
from training_step_check import build_loss_function, build_optimizer
from one_epoch_training_check import run_one_train_epoch, run_validation_epoch

def save_json(data, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)

def build_experiment_report(max_batches, train_loss, val_loss, val_accuracy):
    created_at = datetime.now().isoformat()
    report = {
        "created_at": created_at,
        "max_batches": max_batches,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "interpretation": "This is a limited baseline training experiment, not final model quality.",
    }

    return report

def run_baseline_experiment(max_batches):
    tensor_transform = transforms.ToTensor()
    datasets = build_split_datasets("data/processed", transform=tensor_transform)
    loaders = build_split_loaders(datasets, batch_size=8)

    model = build_baseline_cnn(num_classes=7)
    loss_function = build_loss_function()
    optimizer = build_optimizer(model, 0.001)
    
    train_loss = run_one_train_epoch(model, loaders["train"], loss_function, optimizer, max_batches=max_batches)
    val_loss, val_accuracy = run_validation_epoch(model, loaders["val"], loss_function, max_batches=max_batches)
    report = build_experiment_report(max_batches, train_loss, val_loss, val_accuracy)
    return report

def main():
    max_batches = 100

    report = run_baseline_experiment(max_batches)
    save_json(report, "baseline_experiment_report.json")
    print(report)
if __name__ == "__main__":
    main()


