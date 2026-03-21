#!/usr/bin/env python3
"""
Train an Audio Spectrogram Transformer (AST) to classify between real and AI-generated music.
Converts WAV files to mel spectrograms and trains a vision transformer.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as T
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


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
    real_dir = "data/real"
    ai_dir = "data/ai"
    sample_rate = 16000  # AST expects 16kHz
    segment_duration = 10.0  # seconds
    n_mels = 128
    n_fft = 512
    hop_length = 160
    max_length = 1024  # frames (10.24 seconds at 16kHz with hop 160)

    # Model
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    num_labels = 2  # real vs AI

    # Training
    batch_size = 32
    num_epochs = 10
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_steps = 500
    val_split = 0.1
    test_split = 0.1

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


# Dataset class
class MusicDataset(Dataset):
    def __init__(
        self,
        real_dir,
        ai_dir,
        feature_extractor,
        segment_duration=10.0,
        augment=False,
        config=None,
    ):
        self.real_dir = real_dir
        self.ai_dir = ai_dir
        self.feature_extractor = feature_extractor
        self.segment_duration = segment_duration
        self.augment = augment
        self.config = config or Config()

        # Collect all audio files
        self.file_paths = []
        self.labels = []

        # Real music (label 0)
        for file in os.listdir(real_dir):
            if file.endswith((".wav", ".WAV", ".mp3", ".flac")):
                self.file_paths.append(os.path.join(real_dir, file))
                self.labels.append(0)

        # AI music (label 1)
        for file in os.listdir(ai_dir):
            if file.endswith((".wav", ".WAV", ".mp3", ".flac")):
                self.file_paths.append(os.path.join(ai_dir, file))
                self.labels.append(1)

        real_count = sum(1 for label in self.labels if label == 0)
        ai_count = sum(1 for label in self.labels if label == 1)
        print(
            f"Dataset: {len(self.file_paths)} files ({real_count} real, {ai_count} AI)"
        )

        # Audio transforms
        self.resampler = T.Resample(orig_freq=44100, new_freq=self.config.sample_rate)

        # SpecAugment transforms
        if self.augment:
            self.freq_mask = T.FrequencyMasking(
                freq_mask_param=self.config.freq_mask_param
            )
            self.time_mask = T.TimeMasking(time_mask_param=self.config.time_mask_param)

    def __len__(self):
        return len(self.file_paths)

    def load_and_segment(self, file_path):
        """Load audio file and extract a random segment of fixed duration."""
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a silent segment
            waveform = torch.zeros(
                1, int(self.config.sample_rate * self.segment_duration)
            )
            sr = self.config.sample_rate

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.config.sample_rate)
            waveform = resampler(waveform)

        # Calculate segment length in samples
        segment_samples = int(self.segment_duration * self.config.sample_rate)

        # If waveform is shorter than segment, pad it
        if waveform.shape[1] < segment_samples:
            padding = segment_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Random start position for training, center for validation
            if self.augment:
                max_start = waveform.shape[1] - segment_samples
                start = random.randint(0, max_start)
            else:
                start = (waveform.shape[1] - segment_samples) // 2
            waveform = waveform[:, start : start + segment_samples]

        return waveform.squeeze().numpy()  # Convert to 1D numpy array

    def apply_specaugment(self, spectrogram_tensor):
        """Apply SpecAugment to a spectrogram tensor (time, freq)."""
        # SpecAugment expects (channel, time, freq) or (time, freq)
        # Our spectrogram is (time, freq) -> add channel dim
        spec = spectrogram_tensor.unsqueeze(0)  # (1, time, freq)

        for _ in range(self.config.num_freq_masks):
            spec = self.freq_mask(spec)
        for _ in range(self.config.num_time_masks):
            spec = self.time_mask(spec)

        return spec.squeeze(0)  # back to (time, freq)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load and segment audio
        waveform = self.load_and_segment(file_path)

        # Extract mel spectrogram using AST feature extractor
        # The feature extractor expects raw waveform and returns log-mel spectrogram
        inputs = self.feature_extractor(
            waveform, sampling_rate=self.config.sample_rate, return_tensors="pt"
        )

        # Get the spectrogram (shape: 1, 1024, 128)
        spectrogram = inputs.input_values.squeeze(0)  # (1024, 128)

        # Apply SpecAugment if in training mode
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
            ignore_mismatched_sizes=True,  # In case the pretrained model has different num_labels
        )

        # Freeze early layers if desired (optional)
        # for param in self.ast.ast.embeddings.parameters():
        #     param.requires_grad = False

    def forward(self, input_values, attention_mask=None):
        outputs = self.ast(input_values=input_values, attention_mask=attention_mask)
        return outputs.logits


# SpecAugment helper (alternative implementation)
def spec_augment(
    spectrogram,
    freq_mask_param=27,
    time_mask_param=70,
    num_freq_masks=2,
    num_time_masks=2,
):
    """Apply SpecAugment to a spectrogram tensor."""
    spec = spectrogram.clone()
    _, time_steps, freq_bins = spec.shape

    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, freq_bins - f)
        spec[:, :, f0 : f0 + f] = 0

    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, time_steps - t)
        spec[:, t0 : t0 + t, :] = 0

    return spec


# Training function
def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (spectrograms, labels) in enumerate(pbar):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(spectrograms)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{accuracy_score(all_labels, all_preds):.4f}",
            }
        )

    avg_loss = total_loss / len(dataloader)
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
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of AI class

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    auc = roc_auc_score(all_labels, all_probs) if len(all_labels) > 1 else 0.5

    return avg_loss, accuracy, precision, recall, f1, auc


# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Train AST for real vs AI music classification"
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        default="data/real",
        help="Directory containing real music WAV files",
    )
    parser.add_argument(
        "--ai_dir",
        type=str,
        default="data/ai",
        help="Directory containing AI-generated music WAV files",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Set seed
    set_seed(42)

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize feature extractor
    print("Loading AST feature extractor...")
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        Config.model_name,
        sampling_rate=Config.sample_rate,
        num_mel_bins=Config.n_mels,
        max_length=Config.max_length,
    )

    # Create datasets
    print("Creating datasets...")
    full_dataset = MusicDataset(
        real_dir=args.real_dir,
        ai_dir=args.ai_dir,
        feature_extractor=feature_extractor,
        segment_duration=Config.segment_duration,
        augment=False,  # No augmentation for splitting
        config=Config,
    )

    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(Config.val_split * dataset_size)
    test_size = int(Config.test_split * dataset_size)
    train_size = dataset_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create datasets with augmentation for training
    train_dataset_augmented = MusicDataset(
        real_dir=args.real_dir,
        ai_dir=args.ai_dir,
        feature_extractor=feature_extractor,
        segment_duration=Config.segment_duration,
        augment=True,  # SpecAugment enabled
        config=Config,
    )

    # Replace the training subset with augmented version
    # We need to create a new dataset with only the training indices
    train_indices = train_dataset.indices
    train_dataset_subset = torch.utils.data.Subset(
        train_dataset_augmented, train_indices
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(
        f"Train: {len(train_dataset_subset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Initialize model
    print("Loading AST model...")
    model = ASTBinaryClassifier(
        model_name=Config.model_name, num_labels=Config.num_labels
    ).to(Config.device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=Config.weight_decay
    )

    # Learning rate scheduler (cosine with warmup)
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0
    )

    # Resume from checkpoint if specified
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

    # Training loop
    print(f"\nStarting training on {Config.device}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, Config.device, scheduler
        )

        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(
            model, val_loader, criterion, Config.device
        )

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(
            f"Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}"
        )

        # Save checkpoint
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
        }
        torch.save(
            checkpoint,
            os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                checkpoint, os.path.join(args.checkpoint_dir, Config.best_model_path)
            )
            print(f"New best model saved with val acc: {val_acc:.4f}")

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final evaluation on test set...")

    # Load best model
    best_model_path = os.path.join(args.checkpoint_dir, Config.best_model_path)
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=Config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")

    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = validate(
        model, test_loader, criterion, Config.device
    )

    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
    print(f"F1 Score: {test_f1:.4f}, AUC: {test_auc:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "final_model.pth"))
    print(f"\nTraining complete! Models saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
