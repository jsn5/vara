"""
Export Conditional Sketch-RNN to ONNX for web inference.

Exports three separate models:
1. TextEncoder - encodes Malayalam text to embedding
2. DecoderInit - initializes LSTM hidden state from z and text
3. DecoderStep - single step of autoregressive generation
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.sketch_rnn import (
    ConditionalSketchRNN,
    MalaYalamCharEncoder,
    SketchDecoder
)


class TextEncoderONNX(nn.Module):
    """Wrapper for text encoder export."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, text_ids):
        """
        Args:
            text_ids: [batch, seq_len] - padded character indices
        Returns:
            text_encoding: [batch, text_dim]
        """
        # For ONNX, we don't use packed sequences (no dynamic lengths)
        embedded = self.text_encoder.embedding(text_ids)
        _, (hidden, _) = self.text_encoder.lstm(embedded)

        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        text_encoding = torch.cat([hidden_fwd, hidden_bwd], dim=-1)

        return text_encoding


class DecoderInitONNX(nn.Module):
    """Wrapper for decoder initialization."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z, text_encoding):
        """
        Args:
            z: [batch, latent_dim]
            text_encoding: [batch, text_dim]
        Returns:
            h0: [num_layers, batch, hidden_dim]
            c0: [num_layers, batch, hidden_dim]
        """
        h0, c0 = self.decoder.init_hidden(z, text_encoding)
        return h0, c0


class DecoderStepONNX(nn.Module):
    """Wrapper for single decoder step."""

    def __init__(self, decoder):
        super().__init__()
        self.lstm = decoder.lstm
        self.mdn = decoder.mdn
        self.num_mixtures = decoder.mdn.num_mixtures

    def forward(self, stroke, z, text_encoding, h, c):
        """
        Single step of decoding.

        Args:
            stroke: [batch, 1, 5] - previous stroke
            z: [batch, latent_dim]
            text_encoding: [batch, text_dim]
            h: [num_layers, batch, hidden_dim] - hidden state
            c: [num_layers, batch, hidden_dim] - cell state

        Returns:
            pi_logits: [batch, num_mixtures]
            mu_x: [batch, num_mixtures]
            mu_y: [batch, num_mixtures]
            sigma_x: [batch, num_mixtures]
            sigma_y: [batch, num_mixtures]
            rho: [batch, num_mixtures]
            pen_logits: [batch, 3]
            h_new: [num_layers, batch, hidden_dim]
            c_new: [num_layers, batch, hidden_dim]
        """
        # Expand z and text for concatenation
        z_exp = z.unsqueeze(1)  # [batch, 1, latent_dim]
        text_exp = text_encoding.unsqueeze(1)  # [batch, 1, text_dim]

        # Concatenate inputs
        lstm_input = torch.cat([stroke, z_exp, text_exp], dim=-1)

        # LSTM step
        lstm_out, (h_new, c_new) = self.lstm(lstm_input, (h, c))

        # MDN output
        mdn_out = self.mdn(lstm_out)

        # Flatten outputs (remove seq dimension)
        pi_logits = mdn_out['pi_logits'].squeeze(1)
        mu_x = mdn_out['mu_x'].squeeze(1)
        mu_y = mdn_out['mu_y'].squeeze(1)
        sigma_x = mdn_out['sigma_x'].squeeze(1)
        sigma_y = mdn_out['sigma_y'].squeeze(1)
        rho = mdn_out['rho'].squeeze(1)
        pen_logits = mdn_out['pen_logits'].squeeze(1)

        return pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho, pen_logits, h_new, c_new


def export_models(checkpoint_path: str, output_dir: str, opset_version: int = 14):
    """Export all model components to ONNX."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    vocab_size = checkpoint['vocab_size']
    print(f"Vocab size: {vocab_size}")

    # Create model with same architecture
    model = ConditionalSketchRNN(
        vocab_size=vocab_size,
        text_embed_dim=128,
        text_hidden_dim=256,
        stroke_hidden_dim=512,
        latent_dim=128,
        decoder_hidden_dim=1024,
        num_mixtures=20,
        dropout=0.0  # Disable dropout for inference
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded. Text dim: {model.text_dim}, Latent dim: {model.latent_dim}")

    # Export metadata
    metadata = {
        'vocab_size': vocab_size,
        'text_dim': model.text_dim,  # 512
        'latent_dim': model.latent_dim,  # 128
        'hidden_dim': model.decoder.hidden_dim,  # 1024
        'num_mixtures': model.decoder.mdn.num_mixtures,  # 20
        'num_layers': model.decoder.num_layers,  # 1
    }

    # Save metadata as JSON
    import json
    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved config to {output_dir / 'model_config.json'}")

    # 1. Export Text Encoder
    print("\nExporting TextEncoder...")
    text_encoder = TextEncoderONNX(model.text_encoder)
    text_encoder.eval()

    dummy_text = torch.randint(0, vocab_size, (1, 20))  # batch=1, max_len=20

    torch.onnx.export(
        text_encoder,
        dummy_text,
        str(output_dir / 'text_encoder.onnx'),
        input_names=['text_ids'],
        output_names=['text_encoding'],
        dynamic_axes={
            'text_ids': {0: 'batch', 1: 'seq_len'},
            'text_encoding': {0: 'batch'}
        },
        opset_version=opset_version
    )
    print(f"  Saved to {output_dir / 'text_encoder.onnx'}")

    # 2. Export Decoder Init
    print("\nExporting DecoderInit...")
    decoder_init = DecoderInitONNX(model.decoder)
    decoder_init.eval()

    dummy_z = torch.randn(1, model.latent_dim)
    dummy_text_enc = torch.randn(1, model.text_dim)

    torch.onnx.export(
        decoder_init,
        (dummy_z, dummy_text_enc),
        str(output_dir / 'decoder_init.onnx'),
        input_names=['z', 'text_encoding'],
        output_names=['h0', 'c0'],
        dynamic_axes={
            'z': {0: 'batch'},
            'text_encoding': {0: 'batch'},
            'h0': {1: 'batch'},
            'c0': {1: 'batch'}
        },
        opset_version=opset_version
    )
    print(f"  Saved to {output_dir / 'decoder_init.onnx'}")

    # 3. Export Decoder Step
    print("\nExporting DecoderStep...")
    decoder_step = DecoderStepONNX(model.decoder)
    decoder_step.eval()

    dummy_stroke = torch.zeros(1, 1, 5)
    dummy_stroke[0, 0, 2] = 1  # pen down
    dummy_h = torch.zeros(model.decoder.num_layers, 1, model.decoder.hidden_dim)
    dummy_c = torch.zeros(model.decoder.num_layers, 1, model.decoder.hidden_dim)

    torch.onnx.export(
        decoder_step,
        (dummy_stroke, dummy_z, dummy_text_enc, dummy_h, dummy_c),
        str(output_dir / 'decoder_step.onnx'),
        input_names=['stroke', 'z', 'text_encoding', 'h', 'c'],
        output_names=['pi_logits', 'mu_x', 'mu_y', 'sigma_x', 'sigma_y',
                      'rho', 'pen_logits', 'h_new', 'c_new'],
        dynamic_axes={
            'stroke': {0: 'batch'},
            'z': {0: 'batch'},
            'text_encoding': {0: 'batch'},
            'h': {1: 'batch'},
            'c': {1: 'batch'},
            'pi_logits': {0: 'batch'},
            'mu_x': {0: 'batch'},
            'mu_y': {0: 'batch'},
            'sigma_x': {0: 'batch'},
            'sigma_y': {0: 'batch'},
            'rho': {0: 'batch'},
            'pen_logits': {0: 'batch'},
            'h_new': {1: 'batch'},
            'c_new': {1: 'batch'}
        },
        opset_version=opset_version
    )
    print(f"  Saved to {output_dir / 'decoder_step.onnx'}")

    # Verify exports
    print("\nVerifying ONNX exports...")
    import onnx

    for name in ['text_encoder.onnx', 'decoder_init.onnx', 'decoder_step.onnx']:
        path = output_dir / name
        model_onnx = onnx.load(str(path))
        onnx.checker.check_model(model_onnx)
        print(f"  {name}: OK")

    print(f"\n✓ All models exported to {output_dir}")
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Sketch-RNN to ONNX')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='onnx_models',
                        help='Output directory for ONNX models')
    parser.add_argument('--opset', type=int, default=14,
                        help='ONNX opset version')

    args = parser.parse_args()
    export_models(args.checkpoint, args.output, args.opset)
