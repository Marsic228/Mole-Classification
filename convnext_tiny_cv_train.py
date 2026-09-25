import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from sklearn.metrics import f1_score

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)

from cross_validation_split import get_fold_metadata
from cross_validation_dataset import CrossValidationDataset

from utils import count_images_per_class, build_class_weights
from train import save_json, save_checkpoint, build_epoch_report

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()

TRAINABLE_FEATURE_MODULES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

def run_validation_epoch_with_macro_f1(
    model,
    val_loader,
    loss_function,
    device
):
    model.eval()

    losses = []
    all_predictions = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = loss_function(logits, labels)
            losses.append(loss.item())

            predictions = logits.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )
            all_labels.extend(
                labels.cpu().tolist()
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    val_loss = sum(losses) / len(losses)
    val_accuracy = correct / total

    val_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(7)),
        average="macro",
        zero_division=0
    )

    return val_loss, val_accuracy, val_macro_f1


def run_convnext_train_epoch(
    model,
    train_loader,
    loss_function,
    optimizer,
    scaler,
    device
):
    model.eval()

    model.features[-TRAINABLE_FEATURE_MODULES:].train()
    model.classifier.train()

    losses = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        loss = run_training_step_amp(
            model,
            images,
            labels,
            loss_function,
            optimizer,
            scaler
        )

        losses.append(loss)

    return sum(losses) / len(losses)

def run_training_step_amp(
    model,
    images,
    labels,
    loss_function,
    optimizer,
    scaler
):
    optimizer.zero_grad()

    with torch.amp.autocast("cuda"):
        logits = model(images)
        loss = loss_function(logits, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

def run_fold(fold_number, num_epochs=15):
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    transform = weights.transforms()

    train_metadata, val_metadata = get_fold_metadata(fold_number)

    train_dataset = CrossValidationDataset(
        train_metadata,
        "data/processed",
        transform=transform
    )

    val_dataset = CrossValidationDataset(
        val_metadata,
        "data/processed",
        transform=transform
    )

    class_counts = count_images_per_class(train_dataset)
    class_weights = build_class_weights(class_counts)
    sample_weights = class_weights[train_dataset.targets]

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )

    model = convnext_tiny(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.features[-TRAINABLE_FEATURE_MODULES:].parameters():
        parameter.requires_grad = True

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(
        in_features,
        7
    )

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    model = model.to(DEVICE)

    loss_function = FocalLoss(gamma=2.0)


    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=1e-4
    )
    scaler = torch.amp.GradScaler("cuda")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )

    history = []

    best_macro_f1 = -float("inf")
    best_epoch = None

    patience = 4
    epochs_without_improvement = 0

    

    for epoch in range(1, num_epochs + 1):
        train_loss = run_convnext_train_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            scaler,
            DEVICE
        )

        val_loss, val_accuracy, val_macro_f1 = (
            run_validation_epoch_with_macro_f1(
                model,
                val_loader,
                loss_function,
                DEVICE
            )
        )

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_report = build_epoch_report(
            epoch,
            train_loss,
            val_loss,
            val_accuracy
        )

        epoch_report["val_macro_f1"] = val_macro_f1
        epoch_report["learning_rate"] = current_lr

        history.append(epoch_report)

        print(
            f"Fold {fold_number}: "
            f"epoch={epoch}, "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}, "
            f"val_macro_f1={val_macro_f1:.4f}, "
            f"lr={current_lr:.8f}"
        )

        
        save_json(
            epoch_report,
            f"reports/convnext_tiny_finetune_v4_gpu_amp_"
            f"fold{fold_number}_epoch_{epoch}_report.json"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                epoch_report,
                f"checkpoints/convnext_tiny_finetune_v4_gpu_amp_"
                f"fold{fold_number}_best.pt"
            )

            print(
                f"  -> New best checkpoint: "
                f"macro F1={best_macro_f1:.4f}"
            )
        
        
        else:
            epochs_without_improvement += 1

            print(
                f"  -> No improvement "
                f"({epochs_without_improvement}/{patience})"
            )
        scheduler.step()

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping fold {fold_number}. "
                f"Best epoch={best_epoch}, "
                f"best macro F1={best_macro_f1:.4f}"
            )
            break

    save_json(
        history,
        f"reports/convnext_tiny_finetune_v4_gpu_amp_"
        f"fold{fold_number}_training_history.json"
    )

    best_summary = {
        "fold": fold_number,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
    }

    save_json(
        best_summary,
        f"reports/convnext_tiny_finetune_v4_gpu_amp_"
        f"fold{fold_number}_best_summary.json"
    )

    return history

if __name__ == "__main__":
    start_time = time.perf_counter()

    for fold_number in range(1, 6):
        print(f"\n===== FOLD {fold_number} =====")
        run_fold(fold_number, num_epochs=15)

    elapsed = time.perf_counter() - start_time

    print(
        f"\nFull v4 GPU+AMP CV time: "
        f"{elapsed / 60:.1f} minutes"
    )