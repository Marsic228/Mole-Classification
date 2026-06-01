from pathlib import Path
from PIL import Image
from validate_image_paths import load_metadata
from split_config import build_split_config
from split_dataset import group_records_by_lesion_id, split_lesion_ids, assign_records_to_splits
from preprocessing_config import load_preprocessing_config
import json

def build_output_path(output_dir, split_name, label, image_id, config):
    width = config["resize_width"]
    height = config["resize_height"]
    output_path = Path(output_dir)/f"{split_name}"/f"{label}"/f"{image_id}_{width}x{height}.jpg"
    return output_path

def ensure_output_folder(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

def resize_and_save_image(input_path, output_path, config):
    input_path = Path(input_path)
    output_path = Path(output_path)
    image = Image.open(input_path)
    target_size = (config["resize_width"], config["resize_height"])
    resized_image = image.resize(target_size)
    ensure_output_folder(output_path)
    resized_image.save(output_path)
    return output_path

def preprocess_one_record(record, split_name, image_dir, output_dir, config):
    image_id = record["image_id"]
    label = record["dx"]
    input_path = Path(image_dir)/f"{image_id}.jpg"
    output_path = build_output_path(output_dir, split_name, label, image_id, config)
    saved_path = resize_and_save_image(input_path, output_path, config)
    result = {
        "image_id":image_id,
        "split": split_name,
        "label": label,
        "saved_path": str(saved_path)
    }
    return result

def preprocess_split_sample(records, split_name, image_dir, output_dir, config, limit):
    result = []
    for record in records:
        if len(result) >= limit:
            break
        preprocessed_record = preprocess_one_record(record, split_name, image_dir, output_dir, config)
        result.append(preprocessed_record)
    return result

def preprocess_all_split_samples(splits, image_dir, output_dir, config, limit_per_split):
    result = {}
    for split_name in ["train", "val", "test"]:
        records = splits[split_name]
        preprocess_split = preprocess_split_sample(records, split_name, image_dir, output_dir, config, limit_per_split)
        result[split_name] = preprocess_split
    return result

def save_preprocessing_manifest(manifest, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(manifest, file, indent=2)

def build_preprocessing_sample_report(manifest):
    train_count = len(manifest["train"])
    val_count = len(manifest["val"])
    test_count = len(manifest["test"])
    total_processed = train_count + val_count + test_count
    result = {
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "total_processed": total_processed
    }
    return result

def save_preprocessing_report(report, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(report, file, indent=2)
        

if __name__ == "__main__":
    records = load_metadata("HAM10000_metadata.csv")

    split_config = build_split_config()
    grouped = group_records_by_lesion_id(records)
    lesion_ids = list(grouped.keys())
    split_ids = split_lesion_ids(lesion_ids, split_config)
    splits = assign_records_to_splits(grouped, split_ids)

    image_dir = "data/images"
    output_dir = "data/processed_sample_real"
    preprocessing_config = load_preprocessing_config("preprocessing_config.json")
    limit_per_split = 2

    preprocessing = preprocess_all_split_samples(
        splits,
        image_dir,
        output_dir,
        preprocessing_config,
        limit_per_split
    )
    save_preprocessing_manifest(preprocessing, "preprocessing_sample_manifest.json")

    report = build_preprocessing_sample_report(preprocessing)
    save_preprocessing_report(report, "preprocessing_sample_report.json")

