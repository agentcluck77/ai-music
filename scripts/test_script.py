#!/usr/bin/env python3
"""
Smoke-test the AST pipeline against the downloaded SONICS dataset.
Loads small samples from the official train/valid splits, runs a few optimization steps,
and performs one inference pass on a held-out file.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor

from inference import MusicClassifier
from train_ast import (
    ASTBinaryClassifier,
    Config,
    MusicDataset,
    SplitPathResolver,
    set_seed,
)


def load_records_for_split(metadata_csv, split_name, target_count, resolver):
    records = []
    with open(metadata_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split_name:
                continue

            if row["label"] == "real":
                lookup_row = {
                    "label": "real",
                    "filename": row["filename"],
                    "youtube_id": row["youtube_id"],
                    "filepath": f"real_songs/{row['filename']}.mp3",
                }
                binary = 0
            else:
                lookup_row = {
                    "label": row["label"],
                    "filename": row["filename"],
                    "filepath": f"fake_songs/{row['filename']}.mp3",
                }
                binary = 1

            resolved = resolver.resolve(lookup_row)
            if resolved is None:
                continue

            records.append((resolved, binary))
            if len(records) >= target_count:
                break

    if len(records) < target_count:
        raise RuntimeError(
            f"Requested {target_count} rows from {metadata_csv} split={split_name}, found {len(records)}"
        )
    return records


def load_smoke_datasets(args, feature_extractor):
    resolver = SplitPathResolver(args.data_root, args.real_dir, args.ai_dir)
    train_real = args.train_samples // 2
    train_fake = args.train_samples - train_real
    valid_real = args.valid_samples // 2
    valid_fake = args.valid_samples - valid_real

    train_records = load_records_for_split(
        f"{args.data_root}/real_songs.csv", "train", train_real, resolver
    ) + load_records_for_split(
        f"{args.data_root}/fake_songs.csv", "train", train_fake, resolver
    )
    valid_records = load_records_for_split(
        f"{args.data_root}/real_songs.csv", "valid", valid_real, resolver
    ) + load_records_for_split(
        f"{args.data_root}/fake_songs.csv", "valid", valid_fake, resolver
    )

    train_dataset = MusicDataset(
        train_records,
        feature_extractor=feature_extractor,
        segment_duration=Config.segment_duration,
        augment=True,
        config=Config,
    )
    valid_dataset = MusicDataset(
        valid_records,
        feature_extractor=feature_extractor,
        segment_duration=Config.segment_duration,
        augment=False,
        config=Config,
    )
    return train_dataset, valid_dataset, valid_records



def test_dataset_loading(train_dataset):
    print("\nTesting dataset loading...")
    spectrogram, label = train_dataset[0]
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Label: {label}")



def test_training_loop(train_dataset, valid_dataset, batch_size):
    print("\nTesting training loop...")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ASTBinaryClassifier(model_name=Config.model_name, num_labels=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch_count, (spectrograms, labels) in enumerate(train_loader, start=1):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(spectrograms)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_count >= 2:
            break

    avg_loss = total_loss / max(batch_count, 1)
    print(f"Test training loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        for spectrograms, labels in val_loader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            logits = model(spectrograms)
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean()
            print(f"Test validation accuracy: {accuracy:.4f}")
            break

    print("Training loop test passed!")
    return model



def test_inference(model_path, sample_audio_path):
    print("\nTesting inference...")
    classifier = MusicClassifier(model_path)
    result = classifier.classify_file(sample_audio_path)
    print(f"Inference result: {result}")



def main():
    parser = argparse.ArgumentParser(description="Smoke-test AST pipeline on SONICS")
    parser.add_argument("--data_root", type=str, default=Config.data_root)
    parser.add_argument("--real_dir", type=str, default=Config.real_dir)
    parser.add_argument("--ai_dir", type=str, default=Config.ai_dir)
    parser.add_argument("--train_samples", type=int, default=16)
    parser.add_argument("--valid_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_path", type=str, default="test_model.pth")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing AST Music Classifier Pipeline On SONICS")
    print("=" * 60)

    set_seed(42)

    try:
        feature_extractor = ASTFeatureExtractor.from_pretrained(
            Config.model_name,
            sampling_rate=Config.sample_rate,
            num_mel_bins=Config.n_mels,
            max_length=Config.max_length,
        )
        train_dataset, valid_dataset, valid_records = load_smoke_datasets(args, feature_extractor)
        test_dataset_loading(train_dataset)
        model = test_training_loop(train_dataset, valid_dataset, args.batch_size)

        torch.save(model.state_dict(), args.model_path)
        print(f"Saved test model to {args.model_path}")

        sample_audio_path = valid_records[0][0]
        test_inference(args.model_path, sample_audio_path)

        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)

    except Exception as exc:
        print(f"Test failed with error: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        model_path = Path(args.model_path)
        if model_path.exists():
            model_path.unlink()
            print(f"Removed temporary model {model_path}")
        temp_dir = Path("test_data")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
