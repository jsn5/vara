"""
Conditional Sketch-RNN Model in PyTorch
Generates handwriting conditioned on Malayalam text input

Based on "A Neural Representation of Sketch Drawings" by Ha & Eck (2017)
Extended with text conditioning for Malayalam character/word generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class MalaYalamCharEncoder(nn.Module):
    """
    Character-level encoder for Malayalam text.
    Uses embedding + LSTM to encode variable-length text.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_dim = hidden_dim * 2  # bidirectional

    def forward(self, text_ids, text_lengths=None):
        """
        Args:
            text_ids: [batch, max_seq_len] - character indices
            text_lengths: [batch] - actual lengths for packing

        Returns:
            text_encoding: [batch, hidden_dim * 2]
        """
        embedded = self.embedding(text_ids)  # [batch, seq, embed]

        if text_lengths is not None:
            # Pack for variable length sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)

        # Concatenate forward and backward hidden states
        # hidden: [num_layers * 2, batch, hidden_dim]
        hidden_fwd = hidden[-2]  # last layer forward
        hidden_bwd = hidden[-1]  # last layer backward
        text_encoding = torch.cat([hidden_fwd, hidden_bwd], dim=-1)

        return text_encoding


class SketchEncoder(nn.Module):
    """
    Encoder for sketch strokes using bidirectional LSTM.
    Takes stroke-5 format: [dx, dy, p1, p2, p3]
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Project to latent space (VAE style)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    def forward(self, strokes, stroke_lengths=None):
        """
        Args:
            strokes: [batch, max_len, 5] - stroke sequences
            stroke_lengths: [batch] - actual lengths

        Returns:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
            z: [batch, latent_dim] - sampled latent
        """
        if stroke_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                strokes, stroke_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(strokes)

        # Concatenate bidirectional hidden states
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=-1)

        mu = self.fc_mu(hidden_cat)
        logvar = self.fc_logvar(hidden_cat)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return mu, logvar, z


class MDNOutput(nn.Module):
    """
    Mixture Density Network output layer.
    Outputs parameters for Gaussian mixture model for (dx, dy)
    plus categorical distribution for pen state.
    """

    def __init__(self, input_dim: int, num_mixtures: int = 20):
        super().__init__()

        self.num_mixtures = num_mixtures

        # Output: mixture weights, means, std devs, correlations, pen states
        # For each mixture: pi, mu_x, mu_y, sigma_x, sigma_y, rho
        # Plus 3 pen states (p1, p2, p3)
        output_dim = num_mixtures * 6 + 3

        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq, hidden_dim]

        Returns:
            Dictionary with MDN parameters
        """
        y = self.fc(x)  # [batch, seq, output_dim]

        M = self.num_mixtures

        # Split outputs
        pi_logits = y[..., :M]  # mixture weights (logits)
        mu_x = y[..., M:2*M]
        mu_y = y[..., 2*M:3*M]
        sigma_x = torch.exp(y[..., 3*M:4*M])  # must be positive
        sigma_y = torch.exp(y[..., 4*M:5*M])
        rho = torch.tanh(y[..., 5*M:6*M])  # correlation, must be in (-1, 1)
        pen_logits = y[..., 6*M:]  # pen state logits

        return {
            'pi_logits': pi_logits,
            'mu_x': mu_x,
            'mu_y': mu_y,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'rho': rho,
            'pen_logits': pen_logits
        }


class SketchDecoder(nn.Module):
    """
    Decoder LSTM that generates strokes autoregressively.
    Conditioned on latent z and text encoding.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        text_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 1,
        num_mixtures: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input: previous stroke (5) + latent (latent_dim) + text (text_dim)
        input_dim = 5 + latent_dim + text_dim

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.mdn = MDNOutput(hidden_dim, num_mixtures)

        # Initial hidden state from latent + text
        self.fc_h0 = nn.Linear(latent_dim + text_dim, hidden_dim * num_layers)
        self.fc_c0 = nn.Linear(latent_dim + text_dim, hidden_dim * num_layers)

    def init_hidden(self, z, text_encoding):
        """Initialize LSTM hidden state from latent and text encoding."""
        batch_size = z.size(0)

        # Concatenate latent and text
        combined = torch.cat([z, text_encoding], dim=-1)

        h0 = self.fc_h0(combined)
        c0 = self.fc_c0(combined)

        # Reshape for LSTM: [num_layers, batch, hidden]
        h0 = h0.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = c0.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        return h0, c0

    def forward(self, strokes, z, text_encoding, hidden=None):
        """
        Args:
            strokes: [batch, seq, 5] - input strokes (shifted by 1 for teacher forcing)
            z: [batch, latent_dim] - latent vector
            text_encoding: [batch, text_dim] - text conditioning
            hidden: optional initial hidden state

        Returns:
            mdn_params: dict of MDN parameters
            hidden: final hidden state
        """
        batch_size, seq_len, _ = strokes.size()

        # Expand z and text_encoding for all timesteps
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        text_expanded = text_encoding.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate inputs
        lstm_input = torch.cat([strokes, z_expanded, text_expanded], dim=-1)

        if hidden is None:
            hidden = self.init_hidden(z, text_encoding)

        lstm_out, hidden = self.lstm(lstm_input, hidden)
        mdn_params = self.mdn(lstm_out)

        return mdn_params, hidden


class ConditionalSketchRNN(nn.Module):
    """
    Full conditional Sketch-RNN model.
    Encodes strokes to latent space, conditions on text,
    and decodes to generate strokes.
    """

    def __init__(
        self,
        vocab_size: int,
        text_embed_dim: int = 128,
        text_hidden_dim: int = 256,
        stroke_hidden_dim: int = 512,
        latent_dim: int = 128,
        decoder_hidden_dim: int = 1024,
        num_mixtures: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()

        self.text_encoder = MalaYalamCharEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            hidden_dim=text_hidden_dim,
            dropout=dropout
        )

        self.stroke_encoder = SketchEncoder(
            input_dim=5,
            hidden_dim=stroke_hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout
        )

        text_dim = text_hidden_dim * 2  # bidirectional

        self.decoder = SketchDecoder(
            latent_dim=latent_dim,
            text_dim=text_dim,
            hidden_dim=decoder_hidden_dim,
            num_mixtures=num_mixtures,
            dropout=dropout
        )

        self.latent_dim = latent_dim
        self.text_dim = text_dim

    def forward(self, strokes, stroke_lengths, text_ids, text_lengths):
        """
        Forward pass for training.

        Args:
            strokes: [batch, max_stroke_len, 5]
            stroke_lengths: [batch]
            text_ids: [batch, max_text_len]
            text_lengths: [batch]

        Returns:
            mdn_params: dict of MDN parameters
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        # Encode text
        text_encoding = self.text_encoder(text_ids, text_lengths)

        # Encode strokes
        mu, logvar, z = self.stroke_encoder(strokes, stroke_lengths)

        # Prepare decoder input (shift strokes right, prepend start token)
        # Start token is [0, 0, 1, 0, 0] (no movement, pen down)
        batch_size = strokes.size(0)
        start_token = torch.zeros(batch_size, 1, 5, device=strokes.device)
        start_token[..., 2] = 1  # p1 = 1 (pen down)

        decoder_input = torch.cat([start_token, strokes[:, :-1, :]], dim=1)

        # Decode
        mdn_params, _ = self.decoder(decoder_input, z, text_encoding)

        return mdn_params, mu, logvar

    def sample(
        self,
        text_ids,
        text_lengths=None,
        max_len: int = 200,
        temperature: float = 0.4,
        greedy: bool = False,
        z: torch.Tensor = None
    ):
        """
        Generate strokes conditioned on text.

        Args:
            text_ids: [batch, text_len] or [text_len]
            text_lengths: [batch] or None
            max_len: maximum number of stroke points
            temperature: sampling temperature
            greedy: if True, use argmax instead of sampling
            z: optional latent vector to use

        Returns:
            strokes: [batch, seq_len, 5] generated strokes
        """
        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)

        batch_size = text_ids.size(0)
        device = text_ids.device

        # Encode text
        text_encoding = self.text_encoder(text_ids, text_lengths)

        # Sample or use provided latent
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=device)

        # Initialize hidden state
        hidden = self.decoder.init_hidden(z, text_encoding)

        # Start token
        stroke = torch.zeros(batch_size, 1, 5, device=device)
        stroke[..., 2] = 1  # p1 = 1

        generated = []
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            # Prepare input
            z_exp = z.unsqueeze(1)
            text_exp = text_encoding.unsqueeze(1)
            lstm_input = torch.cat([stroke, z_exp, text_exp], dim=-1)

            # Forward through decoder
            lstm_out, hidden = self.decoder.lstm(lstm_input, hidden)
            mdn_params = self.decoder.mdn(lstm_out)

            # Sample next stroke
            next_stroke = self._sample_from_mdn(mdn_params, temperature, greedy)
            generated.append(next_stroke)

            # Check for end token (p3 = 1)
            done = done | (next_stroke[..., 4] > 0.5).squeeze(1)
            if done.all():
                break

            stroke = next_stroke

        return torch.cat(generated, dim=1)

    def _sample_from_mdn(self, mdn_params, temperature=1.0, greedy=False):
        """Sample a single stroke from MDN parameters."""
        pi_logits = mdn_params['pi_logits'].squeeze(1) / temperature
        mu_x = mdn_params['mu_x'].squeeze(1)
        mu_y = mdn_params['mu_y'].squeeze(1)
        sigma_x = mdn_params['sigma_x'].squeeze(1) * temperature
        sigma_y = mdn_params['sigma_y'].squeeze(1) * temperature
        rho = mdn_params['rho'].squeeze(1)
        pen_logits = mdn_params['pen_logits'].squeeze(1) / temperature

        batch_size = pi_logits.size(0)
        device = pi_logits.device

        # Sample mixture component
        if greedy:
            k = pi_logits.argmax(dim=-1)
        else:
            pi = F.softmax(pi_logits, dim=-1)
            k = Categorical(pi).sample()

        # Get parameters for selected component
        idx = k.unsqueeze(-1)
        mu_x_k = mu_x.gather(-1, idx).squeeze(-1)
        mu_y_k = mu_y.gather(-1, idx).squeeze(-1)
        sigma_x_k = sigma_x.gather(-1, idx).squeeze(-1)
        sigma_y_k = sigma_y.gather(-1, idx).squeeze(-1)
        rho_k = rho.gather(-1, idx).squeeze(-1)

        # Sample from bivariate Gaussian
        if greedy:
            dx = mu_x_k
            dy = mu_y_k
        else:
            # Bivariate normal sampling
            z1 = torch.randn(batch_size, device=device)
            z2 = torch.randn(batch_size, device=device)

            dx = mu_x_k + sigma_x_k * z1
            dy = mu_y_k + sigma_y_k * (rho_k * z1 + torch.sqrt(1 - rho_k**2) * z2)

        # Sample pen state
        if greedy:
            pen_idx = pen_logits.argmax(dim=-1)
        else:
            pen_probs = F.softmax(pen_logits, dim=-1)
            pen_idx = Categorical(pen_probs).sample()

        # Create one-hot pen state
        pen_state = F.one_hot(pen_idx, num_classes=3).float()

        # Combine into stroke
        stroke = torch.stack([dx, dy], dim=-1)
        stroke = torch.cat([stroke, pen_state], dim=-1)

        return stroke.unsqueeze(1)


def reconstruction_loss(mdn_params, target_strokes, mask=None):
    """
    Compute reconstruction loss using negative log-likelihood.

    Args:
        mdn_params: dict with MDN parameters
        target_strokes: [batch, seq, 5] - target strokes
        mask: [batch, seq] - mask for valid positions

    Returns:
        loss: scalar tensor
    """
    pi_logits = mdn_params['pi_logits']
    mu_x = mdn_params['mu_x']
    mu_y = mdn_params['mu_y']
    sigma_x = mdn_params['sigma_x']
    sigma_y = mdn_params['sigma_y']
    rho = mdn_params['rho']
    pen_logits = mdn_params['pen_logits']

    # Target values
    dx = target_strokes[..., 0:1]  # [batch, seq, 1]
    dy = target_strokes[..., 1:2]
    pen_target = target_strokes[..., 2:5]  # [batch, seq, 3]

    # Compute Gaussian log probability for each mixture component
    # Formula for bivariate Gaussian log probability
    norm_x = (dx - mu_x) / sigma_x
    norm_y = (dy - mu_y) / sigma_y

    z = norm_x**2 + norm_y**2 - 2 * rho * norm_x * norm_y
    neg_rho_sq = 1 - rho**2

    # Log probability of (dx, dy) under each Gaussian
    log_gaussian = -0.5 * z / neg_rho_sq - \
                   torch.log(sigma_x) - torch.log(sigma_y) - \
                   0.5 * torch.log(neg_rho_sq) - math.log(2 * math.pi)

    # Weight by mixture probabilities and sum
    log_pi = F.log_softmax(pi_logits, dim=-1)
    log_prob_xy = torch.logsumexp(log_pi + log_gaussian, dim=-1)

    # Pen state loss (cross entropy)
    pen_target_idx = pen_target.argmax(dim=-1)
    pen_loss = F.cross_entropy(
        pen_logits.view(-1, 3),
        pen_target_idx.view(-1),
        reduction='none'
    ).view_as(pen_target_idx)

    # Combined loss
    loss = -log_prob_xy + pen_loss

    if mask is not None:
        loss = loss * mask
        loss = loss.sum() / mask.sum()
    else:
        loss = loss.mean()

    return loss


def kl_loss(mu, logvar, kl_weight=1.0, kl_tolerance=0.2):
    """
    KL divergence loss with optional tolerance (free bits).

    Args:
        mu: [batch, latent_dim]
        logvar: [batch, latent_dim]
        kl_weight: weight for KL term
        kl_tolerance: minimum KL per dimension (free bits)

    Returns:
        loss: scalar tensor
    """
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * (1 + logvar - mu**2 - torch.exp(logvar))

    # Apply tolerance (free bits)
    kl = torch.clamp(kl - kl_tolerance, min=0)

    return kl_weight * kl.sum(dim=-1).mean()


class SketchRNNLoss(nn.Module):
    """Combined loss for Sketch-RNN training."""

    def __init__(self, kl_weight=1.0, kl_tolerance=0.2):
        super().__init__()
        self.kl_weight = kl_weight
        self.kl_tolerance = kl_tolerance

    def forward(self, mdn_params, target_strokes, mu, logvar, mask=None):
        """
        Args:
            mdn_params: dict with MDN parameters
            target_strokes: [batch, seq, 5]
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
            mask: [batch, seq] optional mask

        Returns:
            total_loss, recon_loss, kl_loss
        """
        recon = reconstruction_loss(mdn_params, target_strokes, mask)
        kl = kl_loss(mu, logvar, self.kl_weight, self.kl_tolerance)

        return recon + kl, recon, kl
