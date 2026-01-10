"""
Generate Malayalam handwriting from text using trained Conditional Sketch-RNN.
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.sketch_rnn import ConditionalSketchRNN
from models.dataset import MalayalamVocab


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    args = checkpoint['args']
    vocab_size = checkpoint['vocab_size']

    model = ConditionalSketchRNN(
        vocab_size=vocab_size,
        text_embed_dim=args.get('text_embed_dim', 128),
        text_hidden_dim=args.get('text_hidden_dim', 256),
        stroke_hidden_dim=args.get('stroke_hidden_dim', 512),
        latent_dim=args.get('latent_dim', 128),
        decoder_hidden_dim=args.get('decoder_hidden_dim', 1024),
        num_mixtures=args.get('num_mixtures', 20),
        dropout=0.0  # No dropout during inference
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def strokes_to_points(strokes: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Convert stroke-5 format to absolute coordinates.

    Args:
        strokes: [seq_len, 5] - [dx, dy, p1, p2, p3]
        scale: scaling factor

    Returns:
        points: [seq_len, 3] - [x, y, pen_state]
    """
    points = np.zeros((len(strokes), 3))

    x, y = 0, 0
    for i, stroke in enumerate(strokes):
        dx, dy = stroke[0] * scale, stroke[1] * scale
        x += dx
        y += dy

        # Pen state: 0 = drawing, 1 = pen up, 2 = end
        if stroke[4] > 0.5:
            pen = 2
        elif stroke[3] > 0.5:
            pen = 1
        else:
            pen = 0

        points[i] = [x, y, pen]

    return points


def plot_strokes(points: np.ndarray, title: str = '', save_path: str = None):
    """
    Plot handwriting strokes.

    Args:
        points: [seq_len, 3] - [x, y, pen_state]
        title: plot title (Malayalam text)
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Group into strokes
    current_stroke_x = []
    current_stroke_y = []

    for x, y, pen in points:
        current_stroke_x.append(x)
        current_stroke_y.append(y)

        if pen >= 1:  # Pen up or end
            if len(current_stroke_x) > 0:
                ax.plot(current_stroke_x, current_stroke_y, 'b-', linewidth=2)
            current_stroke_x = []
            current_stroke_y = []

            if pen == 2:  # End of drawing
                break

    # Plot any remaining stroke
    if len(current_stroke_x) > 0:
        ax.plot(current_stroke_x, current_stroke_y, 'b-', linewidth=2)

    ax.set_aspect('equal')
    ax.invert_yaxis()  # Y increases downward
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=20, fontname='Noto Sans Malayalam')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        print(f'Saved to {save_path}')

    plt.show()


def generate_handwriting(
    model,
    vocab: MalayalamVocab,
    text: str,
    device: str = 'cpu',
    temperature: float = 0.4,
    max_len: int = 200,
    scale: float = 50.0
):
    """
    Generate handwriting for given text.

    Args:
        model: trained ConditionalSketchRNN
        vocab: MalayalamVocab instance
        text: Malayalam text to generate
        device: torch device
        temperature: sampling temperature (lower = more deterministic)
        max_len: maximum stroke length
        scale: scaling factor for visualization

    Returns:
        points: numpy array of absolute coordinates
    """
    # Encode text
    text_ids = vocab.encode(text)
    text_tensor = torch.tensor([text_ids], dtype=torch.long, device=device)

    # Generate strokes
    with torch.no_grad():
        strokes = model.sample(
            text_tensor,
            max_len=max_len,
            temperature=temperature,
            greedy=False
        )

    strokes = strokes[0].cpu().numpy()

    # Convert to absolute coordinates
    points = strokes_to_points(strokes, scale)

    return points, strokes


def interpolate_latent(
    model,
    vocab: MalayalamVocab,
    text: str,
    device: str = 'cpu',
    num_samples: int = 5,
    temperature: float = 0.4
):
    """
    Generate multiple samples with different latent vectors.

    Args:
        model: trained model
        vocab: vocabulary
        text: text to generate
        device: torch device
        num_samples: number of variations to generate
        temperature: sampling temperature

    Returns:
        list of (points, strokes) tuples
    """
    text_ids = vocab.encode(text)
    text_tensor = torch.tensor([text_ids], dtype=torch.long, device=device)

    samples = []

    for _ in range(num_samples):
        z = torch.randn(1, model.latent_dim, device=device)

        with torch.no_grad():
            strokes = model.sample(
                text_tensor,
                z=z,
                temperature=temperature,
                max_len=200
            )

        strokes_np = strokes[0].cpu().numpy()
        points = strokes_to_points(strokes_np, scale=50.0)
        samples.append((points, strokes_np))

    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate Malayalam handwriting')

    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--text', type=str, default='മലയാളം',
                        help='Malayalam text to generate')
    parser.add_argument('--temperature', type=float, default=0.4,
                        help='Sampling temperature')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Please train the model first by running: python train.py")
        return

    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, args.device)
    vocab = MalayalamVocab()

    print(f'Generating handwriting for: {args.text}')

    if args.num_samples == 1:
        # Single sample
        points, _ = generate_handwriting(
            model, vocab, args.text,
            device=args.device,
            temperature=args.temperature
        )
        plot_strokes(points, args.text, args.output)
    else:
        # Multiple variations
        samples = interpolate_latent(
            model, vocab, args.text,
            device=args.device,
            num_samples=args.num_samples,
            temperature=args.temperature
        )

        # Plot all samples
        fig, axes = plt.subplots(1, args.num_samples, figsize=(4 * args.num_samples, 4))
        if args.num_samples == 1:
            axes = [axes]

        for i, (points, _) in enumerate(samples):
            ax = axes[i]

            current_stroke_x = []
            current_stroke_y = []

            for x, y, pen in points:
                current_stroke_x.append(x)
                current_stroke_y.append(y)

                if pen >= 1:
                    if len(current_stroke_x) > 0:
                        ax.plot(current_stroke_x, current_stroke_y, 'b-', linewidth=2)
                    current_stroke_x = []
                    current_stroke_y = []

                    if pen == 2:
                        break

            if len(current_stroke_x) > 0:
                ax.plot(current_stroke_x, current_stroke_y, 'b-', linewidth=2)

            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.axis('off')
            ax.set_title(f'Sample {i+1}', fontsize=12)

        plt.suptitle(args.text, fontsize=16)
        plt.tight_layout()

        if args.output:
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f'Saved to {args.output}')

        plt.show()


if __name__ == '__main__':
    main()
