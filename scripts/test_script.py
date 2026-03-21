#!/usr/bin/env python3
"""
Test script to verify the training pipeline works.
Generates dummy audio data and runs a small training loop.
"""

import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path


def generate_dummy_audio(
    duration=30.0, sample_rate=16000, frequency=440.0, noise_level=0.1
):
    """Generate a simple sine wave with noise."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    signal = torch.sin(2 * torch.pi * frequency * t)
    noise = torch.randn_like(signal) * noise_level
    waveform = signal + noise
    return waveform.unsqueeze(0)  # Add channel dimension


def create_test_dataset():
    """Create a small test dataset with dummy audio files."""
    print("Creating test dataset...")

    # Create directories
    real_dir = Path("test_data/real")
    ai_dir = Path("test_data/ai")
    real_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)

    # Generate dummy files
    num_files_per_class = 5

    for i in range(num_files_per_class):
        # Real music (different frequencies)
        freq_real = 220 + i * 100
        waveform_real = generate_dummy_audio(
            duration=15.0 + i * 5.0,  # Varying durations
            sample_rate=16000,
            frequency=freq_real,
            noise_level=0.05,
        )
        real_path = real_dir / f"real_{i}.wav"
        torchaudio.save(str(real_path), waveform_real, 16000)

        # AI music (different frequencies)
        freq_ai = 880 + i * 50
        waveform_ai = generate_dummy_audio(
            duration=12.0 + i * 3.0,
            sample_rate=16000,
            frequency=freq_ai,
            noise_level=0.2,  # More noise for AI
        )
        ai_path = ai_dir / f"ai_{i}.wav"
        torchaudio.save(str(ai_path), waveform_ai, 16000)

    print(f"Created {num_files_per_class} files in each directory")
    return str(real_dir), str(ai_dir)


def test_dataset_loading():
    """Test that the dataset can be loaded and processed."""
    print("\nTesting dataset loading...")

    from train_ast import MusicDataset, Config
    from transformers import ASTFeatureExtractor

    # Create test dataset
    real_dir, ai_dir = create_test_dataset()

    # Initialize feature extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        sampling_rate=16000,
        num_mel_bins=128,
        max_length=1024,
    )

    # Create dataset
    dataset = MusicDataset(
        real_dir=real_dir,
        ai_dir=ai_dir,
        feature_extractor=feature_extractor,
        segment_duration=10.0,
        augment=False,
        config=Config,
    )

    print(f"Dataset size: {len(dataset)}")

    # Try loading one sample
    spectrogram, label = dataset[0]
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Label: {label}")

    return dataset


def test_training_loop():
    """Test a minimal training loop."""
    print("\nTesting training loop...")

    from train_ast import MusicDataset, ASTBinaryClassifier, Config
    from transformers import ASTFeatureExtractor
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split

    # Create test dataset
    real_dir, ai_dir = create_test_dataset()

    # Initialize feature extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        sampling_rate=16000,
        num_mel_bins=128,
        max_length=1024,
    )

    # Create dataset
    dataset = MusicDataset(
        real_dir=real_dir,
        ai_dir=ai_dir,
        feature_extractor=feature_extractor,
        segment_duration=10.0,
        augment=False,
        config=Config,
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train size: {train_size}, Val size: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Initialize model
    model = ASTBinaryClassifier(
        model_name="MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=2
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Using device: {device}")

    # Training loop for 1 epoch
    model.train()
    total_loss = 0
    for batch_idx, (spectrograms, labels) in enumerate(train_loader):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(spectrograms)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx >= 2:  # Just test a few batches
            break

    avg_loss = total_loss / min(3, batch_idx + 1)
    print(f"Test training loss: {avg_loss:.4f}")

    # Validation
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


def test_inference():
    """Test inference on a single file."""
    print("\nTesting inference...")

    from inference import MusicClassifier

    # First, train a quick model
    model = test_training_loop()

    # Save model
    torch.save(model.state_dict(), "test_model.pth")
    print("Saved test model")

    # Test inference on a test file
    test_file = "test_data/real/real_0.wav"
    if os.path.exists(test_file):
        classifier = MusicClassifier("test_model.pth")
        result = classifier.classify_file(test_file)
        print(f"Inference result: {result}")
    else:
        print(f"Test file not found: {test_file}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing AST Music Classifier Pipeline")
    print("=" * 60)

    try:
        # Test dataset loading
        dataset = test_dataset_loading()

        # Test training loop
        model = test_training_loop()

        # Test inference
        test_inference()

        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)

        # Clean up test data
        import shutil

        if os.path.exists("test_data"):
            shutil.rmtree("test_data")
            print("Cleaned up test data")

        if os.path.exists("test_model.pth"):
            os.remove("test_model.pth")
            print("Cleaned up test model")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
