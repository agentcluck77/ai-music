#!/usr/bin/env python3
"""
Train an Audio Spectrogram Transformer (AST) to classify between real and AI-generated music.
Uses the official SONICS train/valid/test CSV splits and resolves them onto the downloaded audio.
"""

import argparse
import csv
import os
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification

warnings.filterwarnings("ignore")
csv.field_size_limit(sys.maxsize)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm", ".mp4")
EXTENSION_PRIORITY = {
    ".wav": 0,
    ".flac": 1,
    ".mp3": 2,
    ".m4a": 3,
    ".ogg": 4,
    ".opus": 5,
    ".webm": 6,
    ".mp4": 7,
}


# Set seeds for reproducibility

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Configuration
class Config:
    # Data
    data_root = "/mnt/aloy/sonics"
    real_dir = "real_songs"
    ai_dir = "fake_songs"
    train_csv = "train.csv"
    valid_csv = "valid.csv"
    test_csv = "test.csv"
    sample_rate = 16000  # AST expects 16kHz
    segment_duration = 10.0  # seconds
    n_mels = 128
    max_length = 1024  # frames (10.24 seconds at 16kHz with hop 160)

    # Model
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    num_labels = 2  # real vs any fake mixture

    # Training
    batch_size = 32
    num_epochs = 10
    learning_rate = 2e-5
    weight_decay = 0.01

    # Augmentation (SpecAugment)
    freq_mask_param = 27  # number of mel bins to mask
    time_mask_param = 70  # number of frames to mask
    num_freq_masks = 2
    num_time_masks = 2

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    checkpoint_dir = "checkpoints"
    best_model_path = "best_model.pth"



def prefer_path(current_path, candidate_path):
    if current_path is None:
        return candidate_path
    current_suffix = Path(current_path).suffix.lower()
    candidate_suffix = Path(candidate_path).suffix.lower()
    return (
        candidate_path
        if EXTENSION_PRIORITY.get(candidate_suffix, 99)
        < EXTENSION_PRIORITY.get(current_suffix, 99)
        else current_path
    )


class SplitPathResolver:
    def __init__(self, data_root, real_dir, ai_dir):
        self.data_root = os.path.abspath(data_root)
        self.real_dir = self._resolve_dir(real_dir)
        self.ai_dir = self._resolve_dir(ai_dir)
        self.real_index = self._build_audio_index(self.real_dir)
        self.ai_index = self._build_audio_index(self.ai_dir)

        print(
            f"Indexed audio files: {len(self.real_index)} real stems in {self.real_dir}, "
            f"{len(self.ai_index)} AI stems in {self.ai_dir}"
        )

    def _resolve_dir(self, directory):
        return directory if os.path.isabs(directory) else os.path.join(self.data_root, directory)

    def _build_audio_index(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Audio directory not found: {directory}")

        index = {}
        for entry in os.scandir(directory):
            if not entry.is_file():
                continue
            suffix = Path(entry.name).suffix.lower()
            if suffix not in AUDIO_EXTENSIONS:
                continue
            stem = Path(entry.name).stem
            index[stem] = prefer_path(index.get(stem), entry.path)
        return index

    def resolve(self, row):
        raw_filepath = row.get("filepath", "")
        if raw_filepath:
            candidate = os.path.join(self.data_root, raw_filepath)
            if os.path.exists(candidate):
                return candidate

        if row.get("label") == "real":
            candidates = [
                row.get("youtube_id"),
                row.get("filename"),
                Path(raw_filepath).stem if raw_filepath else None,
            ]
            for stem in candidates:
                if stem and stem in self.real_index:
                    return self.real_index[stem]
            return None

        candidates = [
            row.get("filename"),
            Path(raw_filepath).stem if raw_filepath else None,
        ]
        for stem in candidates:
            if stem and stem in self.ai_index:
                return self.ai_index[stem]
        return None


def resolve_csv_path(data_root, csv_path):
    return csv_path if os.path.isabs(csv_path) else os.path.join(data_root, csv_path)


def binary_label(source_label):
    return 0 if source_label == "real" else 1


def downsample_records(records, max_samples, seed=42):
    if max_samples is None or len(records) <= max_samples:
        return records

    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for record in records:
        by_label[record[1]].append(record)

    target_real = min(len(by_label[0]), max_samples // 2)
    target_fake = min(len(by_label[1]), max_samples - target_real)

    remaining = max_samples - target_real - target_fake
    if remaining > 0:
        extra_real = min(len(by_label[0]) - target_real, remaining)
        target_real += extra_real
        remaining -= extra_real
    if remaining > 0:
        extra_fake = min(len(by_label[1]) - target_fake, remaining)
        target_fake += extra_fake

    selected = rng.sample(by_label[0], target_real) + rng.sample(by_label[1], target_fake)
    rng.shuffle(selected)
    return selected



def load_split_records(csv_path, resolver, max_samples=None, seed=42):
    records = []
    source_labels = Counter()
    skipped = 0
    missing_examples = []

    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source_label = row["label"]
            source_labels[source_label] += 1
            file_path = resolver.resolve(row)
            if file_path is None:
                skipped += 1
                if len(missing_examples) < 5:
                    missing_examples.append(
                        row.get("filepath")
                        or f"real_songs/{row.get('filename', 'unknown')}"
                    )
                continue
            records.append((file_path, binary_label(source_label)))

    records = downsample_records(records, max_samples=max_samples, seed=seed)
    binary_counts = Counter(label for _, label in records)

    print(
        f"Loaded {Path(csv_path).name}: {len(records)} usable rows "
        f"({binary_counts.get(0, 0)} real, {binary_counts.get(1, 0)} fake), skipped {skipped}"
    )
    print(f"  Source labels: {dict(source_labels)}")
    if missing_examples:
        print(f"  Missing examples: {missing_examples}")

    return records


# Dataset class
class MusicDataset(Dataset):
    def __init__(self, records, feature_extractor, segment_duration=10.0, augment=False, config=None):
        self.records = records
        self.feature_extractor = feature_extractor
        self.segment_duration = segment_duration
        self.augment = augment
        self.config = config or Config
        self.file_paths = [record[0] for record in records]
        self.labels = [record[1] for record in records]

        real_count = sum(1 for label in self.labels if label == 0)
        ai_count = sum(1 for label in self.labels if label == 1)
        print(
            f"Dataset: {len(self.file_paths)} files ({real_count} real, {ai_count} fake/mixed), augment={self.augment}"
        )

        if self.augment:
            self.freq_mask = T.FrequencyMasking(freq_mask_param=self.config.freq_mask_param)
            self.time_mask = T.TimeMasking(time_mask_param=self.config.time_mask_param)

    def __len__(self):
        return len(self.file_paths)

    def load_and_segment(self, file_path):
        """Load audio file and extract a random or centered segment of fixed duration."""
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as exc:
            print(f"Error loading {file_path}: {exc}")
            waveform = torch.zeros(1, int(self.config.sample_rate * self.segment_duration))
            sample_rate = self.config.sample_rate

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != self.config.sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.config.sample_rate)
            waveform = resampler(waveform)

        segment_samples = int(self.segment_duration * self.config.sample_rate)

        if waveform.shape[1] < segment_samples:
            padding = segment_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            if self.augment:
                max_start = waveform.shape[1] - segment_samples
                start = random.randint(0, max_start)
            else:
                start = (waveform.shape[1] - segment_samples) // 2
            waveform = waveform[:, start : start + segment_samples]

        return waveform.squeeze().numpy()

    def apply_specaugment(self, spectrogram_tensor):
        spec = spectrogram_tensor.unsqueeze(0)
        for _ in range(self.config.num_freq_masks):
            spec = self.freq_mask(spec)
        for _ in range(self.config.num_time_masks):
            spec = self.time_mask(spec)
        return spec.squeeze(0)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        waveform = self.load_and_segment(file_path)

        inputs = self.feature_extractor(
            waveform, sampling_rate=self.config.sample_rate, return_tensors="pt"
        )
        spectrogram = inputs.input_values.squeeze(0)

        if self.augment:
            spectrogram = self.apply_specaugment(spectrogram)

        return spectrogram, torch.tensor(label, dtype=torch.long)


# Model wrapper
class ASTBinaryClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.ast(input_values=input_values, attention_mask=attention_mask)
        return outputs.logits


# Training function
def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    for spectrograms, labels in pbar:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(spectrograms)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{accuracy_score(all_labels, all_preds):.4f}",
            }
        )

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for spectrograms, labels in tqdm(dataloader, desc="Validation"):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            logits = model(spectrograms)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return avg_loss, accuracy, precision, recall, f1, auc


# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Train AST for real vs AI music classification using SONICS official splits"
    )
    parser.add_argument("--data_root", type=str, default=Config.data_root, help="SONICS dataset root")
    parser.add_argument(
        "--real_dir",
        type=str,
        default=Config.real_dir,
        help="Directory containing downloaded real audio (relative to data_root or absolute)",
    )
    parser.add_argument(
        "--ai_dir",
        type=str,
        default=Config.ai_dir,
        help="Directory containing extracted fake audio (relative to data_root or absolute)",
    )
    parser.add_argument("--train_csv", type=str, default=Config.train_csv, help="Official SONICS train split CSV")
    parser.add_argument("--valid_csv", type=str, default=Config.valid_csv, help="Official SONICS validation split CSV")
    parser.add_argument("--test_csv", type=str, default=Config.test_csv, help="Official SONICS test split CSV")
    parser.add_argument("--batch_size", type=int, default=Config.batch_size, help="Batch size")
    parser.add_argument("--epochs", type=int, default=Config.num_epochs, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=Config.learning_rate, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=Config.checkpoint_dir,
        help="Directory to save checkpoints",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional cap for quick smoke runs")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Optional cap for quick smoke runs")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Optional cap for quick smoke runs")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Using device: {Config.device}")
    print(f"Data root: {os.path.abspath(args.data_root)}")

    print("Loading AST feature extractor...")
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        Config.model_name,
        sampling_rate=Config.sample_rate,
        num_mel_bins=Config.n_mels,
        max_length=Config.max_length,
    )

    print("Resolving SONICS audio paths...")
    resolver = SplitPathResolver(args.data_root, args.real_dir, args.ai_dir)

    train_csv_path = resolve_csv_path(args.data_root, args.train_csv)
    valid_csv_path = resolve_csv_path(args.data_root, args.valid_csv)
    test_csv_path = resolve_csv_path(args.data_root, args.test_csv)

    print("Loading official split records...")
    train_records = load_split_records(
        train_csv_path, resolver, max_samples=args.max_train_samples, seed=42
    )
    valid_records = load_split_records(
        valid_csv_path, resolver, max_samples=args.max_val_samples, seed=42
    )
    test_records = load_split_records(
        test_csv_path, resolver, max_samples=args.max_test_samples, seed=42
    )

    if not train_records or not valid_records or not test_records:
        raise RuntimeError("One of the splits has no usable audio after path resolution.")

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
    test_dataset = MusicDataset(
        test_records,
        feature_extractor=feature_extractor,
        segment_duration=Config.segment_duration,
        augment=False,
        config=Config,
    )

    pin_memory = Config.device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(valid_dataset)}, Test: {len(test_dataset)}")

    print("Loading AST model...")
    model = ASTBinaryClassifier(model_name=Config.model_name, num_labels=Config.num_labels).to(
        Config.device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Config.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=Config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    print(f"\nStarting training on {Config.device}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, Config.device, scheduler
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(
            model, val_loader, criterion, Config.device
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(
            f"Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "args": vars(args),
        }
        torch.save(
            checkpoint,
            os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint["best_val_acc"] = best_val_acc
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, Config.best_model_path))
            print(f"New best model saved with val acc: {val_acc:.4f}")

    print("\n" + "=" * 60)
    print("Final evaluation on test set...")

    best_model_path = os.path.join(args.checkpoint_dir, Config.best_model_path)
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=Config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")

    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = validate(
        model, test_loader, criterion, Config.device
    )

    print("\nTest Results:")
    print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
    print(f"F1 Score: {test_f1:.4f}, AUC: {test_auc:.4f}")

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "final_model.pth"))
    print(f"\nTraining complete! Models saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
