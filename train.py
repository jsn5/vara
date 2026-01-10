"""
Training script for Conditional Sketch-RNN on Malayalam handwriting data.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add models to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.sketch_rnn import ConditionalSketchRNN, SketchRNNLoss
from models.dataset import (
    MalayalamVocab,
    MalayalamHandwritingDataset,
    collate_fn,
    create_dataloader
)


def get_args():
    parser = argparse.ArgumentParser(description='Train Conditional Sketch-RNN')

    # Data
    parser.add_argument('--data_path', type=str, default='data/all_samples.jsonl',
                        help='Path to training data')
    parser.add_argument('--max_stroke_len', type=int, default=200,
                        help='Maximum stroke sequence length')
    parser.add_argument('--max_text_len', type=int, default=50,
                        help='Maximum text length')

    # Model
    parser.add_argument('--text_embed_dim', type=int, default=128,
                        help='Text embedding dimension')
    parser.add_argument('--text_hidden_dim', type=int, default=256,
                        help='Text encoder hidden dimension')
    parser.add_argument('--stroke_hidden_dim', type=int, default=512,
                        help='Stroke encoder hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent space dimension')
    parser.add_argument('--decoder_hidden_dim', type=int, default=1024,
                        help='Decoder hidden dimension')
    parser.add_argument('--num_mixtures', type=int, default=20,
                        help='Number of MDN mixture components')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9999,
                        help='Learning rate decay per step')
    parser.add_argument('--min_lr', type=float, default=0.00001,
                        help='Minimum learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value')

    # KL annealing
    parser.add_argument('--kl_weight_start', type=float, default=0.01,
                        help='Initial KL weight')
    parser.add_argument('--kl_weight_end', type=float, default=1.0,
                        help='Final KL weight')
    parser.add_argument('--kl_anneal_steps', type=int, default=10000,
                        help='Steps for KL annealing')
    parser.add_argument('--kl_tolerance', type=float, default=0.2,
                        help='KL tolerance (free bits)')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    return parser.parse_args()


def get_kl_weight(step, start, end, anneal_steps):
    """Compute KL weight with linear annealing."""
    if step >= anneal_steps:
        return end
    return start + (end - start) * step / anneal_steps


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, args, global_step):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch in pbar:
        # Move to device
        text_ids = batch['text_ids'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        strokes = batch['strokes'].to(device)
        stroke_lengths = batch['stroke_lengths'].to(device)
        stroke_mask = batch['stroke_mask'].to(device)

        # Forward pass
        mdn_params, mu, logvar = model(strokes, stroke_lengths, text_ids, text_lengths)

        # Compute loss with annealed KL weight
        kl_weight = get_kl_weight(
            global_step,
            args.kl_weight_start,
            args.kl_weight_end,
            args.kl_anneal_steps
        )
        criterion.kl_weight = kl_weight

        loss, recon_loss, kl_loss = criterion(mdn_params, strokes, mu, logvar, stroke_mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Learning rate decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * args.lr_decay, args.min_lr)

        # Accumulate metrics
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1
        global_step += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}',
            'kl_w': f'{kl_weight:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon / num_batches,
        'kl_loss': total_kl / num_batches,
        'global_step': global_step
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    num_batches = 0

    for batch in dataloader:
        text_ids = batch['text_ids'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        strokes = batch['strokes'].to(device)
        stroke_lengths = batch['stroke_lengths'].to(device)
        stroke_mask = batch['stroke_mask'].to(device)

        mdn_params, mu, logvar = model(strokes, stroke_lengths, text_ids, text_lengths)
        loss, recon_loss, kl_loss = criterion(mdn_params, strokes, mu, logvar, stroke_mask)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon / num_batches,
        'kl_loss': total_kl / num_batches
    }


def save_checkpoint(model, optimizer, epoch, global_step, metrics, args, vocab, filename):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'args': vars(args),
        'vocab_size': len(vocab)
    }
    torch.save(checkpoint, filename)
    print(f'Saved checkpoint: {filename}')


def load_checkpoint(filename, model, optimizer=None):
    """Load training checkpoint."""
    checkpoint = torch.load(filename, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['global_step'], checkpoint.get('metrics', {})


def main():
    args = get_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize vocabulary
    vocab = MalayalamVocab()
    print(f'Vocabulary size: {len(vocab)}')

    # Check if data exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please collect some handwriting samples first by running the web app.")
        print("Run: python app.py")
        return

    # Create datasets
    print('Loading training data...')
    train_dataset = MalayalamHandwritingDataset(
        args.data_path,
        vocab,
        max_stroke_len=args.max_stroke_len,
        max_text_len=args.max_text_len,
        normalize=True,
        augment=True
    )

    if len(train_dataset) == 0:
        print("No training samples found. Please collect some data first.")
        return

    # For small datasets, use same data for validation
    # In production, split properly
    val_dataset = MalayalamHandwritingDataset(
        args.data_path,
        vocab,
        max_stroke_len=args.max_stroke_len,
        max_text_len=args.max_text_len,
        normalize=True,
        augment=False
    )
    val_dataset.scale_factor = train_dataset.scale_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')

    # Create model
    device = torch.device(args.device)
    print(f'Using device: {device}')

    model = ConditionalSketchRNN(
        vocab_size=len(vocab),
        text_embed_dim=args.text_embed_dim,
        text_hidden_dim=args.text_hidden_dim,
        stroke_hidden_dim=args.stroke_hidden_dim,
        latent_dim=args.latent_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        num_mixtures=args.num_mixtures,
        dropout=args.dropout
    ).to(device)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = SketchRNNLoss(kl_weight=args.kl_weight_start, kl_tolerance=args.kl_tolerance)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0

    if args.resume:
        print(f'Resuming from {args.resume}')
        start_epoch, global_step, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1

    # Training loop
    best_val_loss = float('inf')
    history = []

    print(f'\nStarting training for {args.epochs} epochs...\n')

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args, global_step
        )
        global_step = train_metrics['global_step']

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Log metrics
        print(f'\nEpoch {epoch}:')
        print(f'  Train - Loss: {train_metrics["loss"]:.4f}, '
              f'Recon: {train_metrics["recon_loss"]:.4f}, '
              f'KL: {train_metrics["kl_loss"]:.4f}')
        print(f'  Val   - Loss: {val_metrics["loss"]:.4f}, '
              f'Recon: {val_metrics["recon_loss"]:.4f}, '
              f'KL: {val_metrics["kl_loss"]:.4f}')

        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics
        })

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, epoch, global_step,
                val_metrics, args, vocab,
                checkpoint_dir / 'best_model.pt'
            )

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, global_step,
                val_metrics, args, vocab,
                checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            )

    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs - 1, global_step,
        val_metrics, args, vocab,
        checkpoint_dir / 'final_model.pt'
    )

    # Save training history
    with open(checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print('\nTraining complete!')
    print(f'Best validation loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()
