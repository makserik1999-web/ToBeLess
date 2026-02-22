"""
SlowFast Fine-Tuning on RWF-2000 Dataset

This script fine-tunes a pre-trained SlowFast R50 model on the RWF-2000
violence detection dataset for 2-class classification (Fight vs NonFight).

Usage:
    python train_slowfast.py --data_dir datasets/RWF-2000 --epochs 20

WARNING: Stop app.py before training to free GPU VRAM!
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from rwf_dataset import RWFDataset, collate_slowfast, create_dataloaders


class SlowFastFineTuner:
    """
    Fine-tuning pipeline for SlowFast on RWF-2000

    Strategy:
    1. Load pre-trained SlowFast R50 from PyTorchVideo
    2. Freeze backbone (blocks) to prevent catastrophic forgetting
    3. Replace final classification head with 2-class output
    4. Train with CrossEntropyLoss + class weights (handle imbalance)
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str = "models/rwf_finetuned",
        batch_size: int = 8,
        num_workers: int = 4,
        learning_rate: float = 0.001,
        epochs: int = 20,
        device: str = "cuda",
        freeze_backbone: bool = True,
        use_amp: bool = True,  # Automatic Mixed Precision
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device if torch.cuda.is_available() else "cpu"
        self.freeze_backbone = freeze_backbone
        self.use_amp = use_amp and self.device == "cuda"

        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if self.use_amp else None
        self.criterion = None

        self.train_loader = None
        self.val_loader = None

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        print("=" * 60)
        print("SlowFast Fine-Tuning on RWF-2000")
        print("=" * 60)
        print(f"  Device: {self.device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {epochs}")
        print(f"  Freeze backbone: {freeze_backbone}")
        print(f"  Mixed Precision (AMP): {self.use_amp}")
        print(f"  Output: {self.output_dir}")
        print("=" * 60)

    def setup(self):
        """Initialize model, dataloaders, optimizer, and criterion"""
        self._setup_model()
        self._setup_dataloaders()
        self._setup_optimizer()
        self._setup_criterion()

    def _setup_model(self):
        """Load and modify SlowFast R50 model"""
        print("\n[1/4] Loading pre-trained SlowFast R50...")

        try:
            from pytorchvideo.models.hub import slowfast_r50

            # Load pre-trained model
            self.model = slowfast_r50(pretrained=True)

            # Freeze backbone if requested
            if self.freeze_backbone:
                print("  Freezing backbone layers...")
                for name, param in self.model.named_parameters():
                    # Only train the head (projection layer)
                    if "head" not in name.lower() and "proj" not in name.lower():
                        param.requires_grad = False

                # Count trainable parameters
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

            # Replace classification head: 400 classes -> 2 classes
            print("  Replacing classification head (400 -> 2 classes)...")

            # SlowFast R50 head structure:
            # model.blocks[6].proj is the final projection layer
            # Input features: 2304 (slow: 2048 + fast: 256)
            in_features = self.model.blocks[6].proj.in_features
            self.model.blocks[6].proj = nn.Linear(in_features, 2)

            print(f"  New head: Linear({in_features} -> 2)")

            # Move to device
            self.model = self.model.to(self.device)
            print(f"  Model moved to {self.device}")

        except Exception as e:
            print(f"\n[ERROR] Failed to load model: {e}")
            raise

    def _setup_dataloaders(self):
        """Create train and validation dataloaders"""
        print("\n[2/4] Setting up dataloaders...")

        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"

        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")

        self.train_loader, self.val_loader = create_dataloaders(
            train_dir=str(train_dir),
            val_dir=str(val_dir),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_frames=32
        )

        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        print("\n[3/4] Setting up optimizer...")

        # Only optimize trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        # AdamW with weight decay
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )

        print(f"  Optimizer: AdamW (lr={self.learning_rate}, weight_decay=0.01)")
        print(f"  Scheduler: CosineAnnealingLR (T_max={self.epochs})")

    def _setup_criterion(self):
        """Setup loss function with class weights"""
        print("\n[4/4] Setting up loss function...")

        # Calculate class weights from training data
        train_dataset = self.train_loader.dataset
        class_counts = [0, 0]  # [NonFight, Fight]
        for _, label in train_dataset.samples:
            class_counts[label] += 1

        # Inverse frequency weighting
        total = sum(class_counts)
        weights = [total / (2 * count) for count in class_counts]
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        print(f"  Class counts: NonFight={class_counts[0]}, Fight={class_counts[1]}")
        print(f"  Class weights: {weights.tolist()}")

        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (clips, labels) in enumerate(self.train_loader):
            # Move to device
            slow = clips[0].to(self.device)
            fast = clips[1].to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model([slow, fast])
                    loss = self.criterion(outputs, labels)

                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model([slow, fast])
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.train_loader):
                acc = 100.0 * correct / total
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)}: "
                      f"Loss={loss.item():.4f}, Acc={acc:.2f}%")

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # Per-class accuracy
        class_correct = [0, 0]
        class_total = [0, 0]

        for clips, labels in self.val_loader:
            slow = clips[0].to(self.device)
            fast = clips[1].to(self.device)
            labels = labels.to(self.device)

            outputs = self.model([slow, fast])
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        # Per-class accuracy
        nf_acc = 100.0 * class_correct[0] / max(1, class_total[0])
        f_acc = 100.0 * class_correct[1] / max(1, class_total[1])

        print(f"  Per-class: NonFight={nf_acc:.2f}%, Fight={f_acc:.2f}%")

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "config": {
                "num_classes": 2,
                "class_names": ["NonFight", "Fight"],
                "freeze_backbone": self.freeze_backbone,
                "learning_rate": self.learning_rate,
            }
        }

        # Save latest
        latest_path = self.output_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  [NEW BEST] Saved to {best_path}")

    def train(self):
        """Full training loop"""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            print("-" * 40)

            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

            # Validate
            val_loss, val_acc = self.validate()
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"  LR: {current_lr:.6f}")

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch

            self.save_checkpoint(epoch, is_best=is_best)

        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"  Best model saved to: {self.output_dir / 'best_model.pth'}")

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  Training history saved to: {history_path}")

        return self.best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SlowFast on RWF-2000")

    # Data
    parser.add_argument("--data_dir", type=str, default="datasets/RWF-2000",
                        help="Path to RWF-2000 dataset")
    parser.add_argument("--output_dir", type=str, default="models/rwf_finetuned",
                        help="Output directory for checkpoints")

    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loading workers")

    # Model
    parser.add_argument("--no_freeze", action="store_true",
                        help="Don't freeze backbone (train entire model)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")

    args = parser.parse_args()

    # Check GPU
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available. Training on CPU will be VERY slow!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(0)

    # Check dataset
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\n[ERROR] Dataset not found: {data_dir}")
        print("\nPlease download RWF-2000 dataset first.")
        print("See datasets/README.md for instructions.")
        sys.exit(1)

    # Create trainer
    trainer = SlowFastFineTuner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        epochs=args.epochs,
        freeze_backbone=not args.no_freeze,
        use_amp=not args.no_amp,
    )

    # Setup and train
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
