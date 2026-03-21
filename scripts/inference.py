#!/usr/bin/env python3
"""
Inference script for AST music classifier.
Loads a trained model and classifies new audio files as real or AI-generated.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import torchaudio.transforms as T
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import argparse
from pathlib import Path


# Model class (same as in training script)
class ASTBinaryClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.ast(input_values=input_values, attention_mask=attention_mask)
        return outputs.logits


class MusicClassifier:
    def __init__(
        self,
        model_path,
        model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_name = model_name

        # Load feature extractor
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            model_name, sampling_rate=16000, num_mel_bins=128, max_length=1024
        )

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Constants
        self.sample_rate = 16000
        self.segment_duration = 10.0  # seconds
        self.segment_samples = int(self.segment_duration * self.sample_rate)

    def _load_model(self, model_path):
        """Load trained model from checkpoint."""
        model = ASTBinaryClassifier(model_name=self.model_name, num_labels=2).to(
            self.device
        )

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)

            # Check if it's a full checkpoint or just state dict
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print(
                    f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})"
                )
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model state dict")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        return model

    def load_audio(self, audio_path):
        """Load audio file and convert to 16kHz mono."""
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            raise ValueError(f"Error loading audio file {audio_path}: {e}")

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze().numpy()  # 1D numpy array

    def extract_segments(self, waveform):
        """Extract multiple overlapping segments from a long audio file."""
        total_samples = len(waveform)

        if total_samples < self.segment_samples:
            # Pad if too short
            padding = self.segment_samples - total_samples
            waveform = np.pad(waveform, (0, padding), mode="constant")
            return [waveform]

        # Calculate number of segments with 50% overlap
        hop_samples = self.segment_samples // 2
        num_segments = (total_samples - self.segment_samples) // hop_samples + 1

        segments = []
        for i in range(num_segments):
            start = i * hop_samples
            end = start + self.segment_samples
            segments.append(waveform[start:end])

        # Ensure we include the last segment if not covered
        if (num_segments - 1) * hop_samples + self.segment_samples < total_samples:
            segments.append(waveform[-self.segment_samples :])

        return segments

    def predict_segment(self, waveform_segment):
        """Predict class probabilities for a single audio segment."""
        # Extract features
        inputs = self.feature_extractor(
            waveform_segment, sampling_rate=self.sample_rate, return_tensors="pt"
        )

        # Move to device
        input_values = inputs.input_values.to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(input_values)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities.cpu().numpy()[0]  # [real_prob, ai_prob]

    def predict_file(self, audio_path, return_segments=False):
        """Predict class probabilities for an entire audio file."""
        # Load audio
        waveform = self.load_audio(audio_path)

        # Extract segments
        segments = self.extract_segments(waveform)

        # Predict each segment
        segment_probs = []
        for segment in segments:
            probs = self.predict_segment(segment)
            segment_probs.append(probs)

        # Average probabilities across segments
        segment_probs = np.array(segment_probs)
        avg_probs = np.mean(segment_probs, axis=0)

        if return_segments:
            return avg_probs, segment_probs
        else:
            return avg_probs

    def classify_file(self, audio_path, threshold=0.5):
        """Classify an audio file as real or AI-generated."""
        probs = self.predict_file(audio_path)
        ai_prob = probs[1]  # Probability of AI class

        if ai_prob > threshold:
            label = "AI-generated"
        else:
            label = "Real"

        confidence = max(probs)

        return {
            "file": audio_path,
            "prediction": label,
            "confidence": float(confidence),
            "ai_probability": float(ai_prob),
            "real_probability": float(probs[0]),
        }

    def batch_classify(self, audio_paths, threshold=0.5):
        """Classify multiple audio files."""
        results = []
        for path in tqdm(audio_paths, desc="Classifying"):
            try:
                result = self.classify_file(path, threshold)
                results.append(result)
            except Exception as e:
                print(f"Error classifying {path}: {e}")
                results.append(
                    {
                        "file": path,
                        "prediction": "Error",
                        "confidence": 0.0,
                        "ai_probability": 0.0,
                        "real_probability": 0.0,
                    }
                )
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Classify audio files as real or AI-generated"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument("--audio_path", type=str, help="Path to audio file to classify")
    parser.add_argument(
        "--audio_dir", type=str, help="Directory containing audio files to classify"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="Save results to JSON file"
    )
    args = parser.parse_args()

    # Initialize classifier
    print("Loading model...")
    classifier = MusicClassifier(args.model_path)

    # Get audio files
    audio_files = []
    if args.audio_path:
        audio_files = [args.audio_path]
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        audio_files = (
            list(audio_dir.glob("*.wav"))
            + list(audio_dir.glob("*.WAV"))
            + list(audio_dir.glob("*.mp3"))
            + list(audio_dir.glob("*.flac"))
        )
    else:
        print("Please specify either --audio_path or --audio_dir")
        return

    if not audio_files:
        print("No audio files found")
        return

    print(f"Found {len(audio_files)} audio files")

    # Classify
    results = []
    for audio_file in audio_files:
        print(f"\nClassifying: {audio_file}")
        result = classifier.classify_file(str(audio_file), args.threshold)
        results.append(result)

        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"AI probability: {result['ai_probability']:.3f}")
        print(f"Real probability: {result['real_probability']:.3f}")

    # Save results if requested
    if args.output_file:
        import json

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Classification Summary:")
    real_count = sum(1 for r in results if r["prediction"] == "Real")
    ai_count = sum(1 for r in results if r["prediction"] == "AI-generated")
    error_count = sum(1 for r in results if r["prediction"] == "Error")

    print(f"Real: {real_count}")
    print(f"AI-generated: {ai_count}")
    if error_count > 0:
        print(f"Errors: {error_count}")


if __name__ == "__main__":
    from tqdm import tqdm  # Import here to avoid circular import

    main()
