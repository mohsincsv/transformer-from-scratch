"""
Transformer model implementation in PyTorch.

This module contains the implementation of a Transformer model as described in
the paper "Attention Is All You Need" by Vaswani et al. (2017).

The code is organized into several classes, each representing a component of
the Transformer architecture.
"""

import torch
import math
import torch.nn as nn


class LayerNormalization(nn.Module):
    """Implements Layer Normalization."""

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        """
        Initialize the Layer Normalization module.

        Args:
            features (int): The number of features in the input.
            eps (float): A small number for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of Layer Normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized output.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """Implements the Feed Forward Block of the Transformer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Initialize the Feed Forward Block.

        Args:
            d_model (int): The model's hidden dimension.
            d_ff (int): The inner dimension of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the Feed Forward Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after feed-forward transformation.
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(nn.Module):
    """Implements input embeddings for the Transformer."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initialize the Input Embeddings module.

        Args:
            d_model (int): The model's hidden dimension.
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the Input Embeddings.

        Args:
            x (torch.Tensor): Input tensor of token indices.

        Returns:
            torch.Tensor: Embedded representation of the input.
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implements positional encoding for the Transformer."""

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Initialize the Positional Encoding module.

        Args:
            d_model (int): The model's hidden dimension.
            seq_len (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Input with added positional encoding.
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    """Implements residual connection for the Transformer."""

    def __init__(self, features: int, dropout: float) -> None:
        """
        Initialize the Residual Connection module.

        Args:
            features (int): Number of features in the input.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        """
        Apply residual connection to any sublayer with the same size.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer (callable): A function implementing a sublayer.

        Returns:
            torch.Tensor: Output after applying the residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    """Implements multi-head attention mechanism."""

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        Initialize the Multi-Head Attention Block.

        Args:
            d_model (int): The model's hidden dimension.
            h (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Attention mask.
            dropout (nn.Dropout): Dropout module.

        Returns:
            tuple: Attention output and attention scores.
        """
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        Perform the forward pass of the Multi-Head Attention Block.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Output after multi-head attention.
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    """Implements a single encoder block of the Transformer."""

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float
    ) -> None:
        """
        Initialize the Encoder Block.

        Args:
            features (int): Number of features in the input.
            self_attention_block (MultiHeadAttentionBlock): Self-attention module.
            feed_forward_block (FeedForwardBlock): Feed-forward module.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(features, dropout) for _ in range(2)
        ])

    def forward(self, x, src_mask):
        """
        Perform the forward pass of the Encoder Block.

        Args:
            x (torch.Tensor): Input tensor.
            src_mask (torch.Tensor): Source mask for attention.

        Returns:
            torch.Tensor: Output after self-attention and feed-forward layers.
        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """Implements the full encoder stack of the Transformer."""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Initialize the Encoder.

        Args:
            features (int): Number of features in the input.
            layers (nn.ModuleList): List of EncoderBlock modules.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """
        Perform the forward pass of the Encoder.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Output after all encoder layers.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """Implements a single decoder block of the Transformer."""

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float
    ) -> None:
        """
        Initialize the Decoder Block.

        Args:
            features (int): Number of features in the input.
            self_attention_block (MultiHeadAttentionBlock): Self-attention module.
            cross_attention_block (MultiHeadAttentionBlock): Cross-attention module.
            feed_forward_block (FeedForwardBlock): Feed-forward module.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(features, dropout) for _ in range(3)
        ])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Perform the forward pass of the Decoder Block.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask for attention.
            tgt_mask (torch.Tensor): Target mask for attention.

        Returns:
            torch.Tensor: Output after self-attention, cross-attention, and feed-forward layers.
        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """Implements the full decoder stack of the Transformer."""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Initialize the Decoder.

        Args:
            features (int): Number of features in the input.
            layers (nn.ModuleList): List of DecoderBlock modules.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Perform the forward pass of the Decoder.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask for attention.
            tgt_mask (torch.Tensor): Target mask for attention.

        Returns:
            torch.Tensor: Output after all decoder layers.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """Implements the final projection layer of the Transformer."""

    def __init__(self, d_model, vocab_size) -> None:
        """
        Initialize the Projection Layer.

        Args:
            d_model (int): The model's hidden dimension.
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> torch.Tensor:
        """
        Perform the forward pass of the Projection Layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Projected output.
        """
        return self.proj(x)


class Transformer(nn.Module):
    """Implements the full Transformer model."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer
    ) -> None:
        """
        Initialize the Transformer model.

        Args:
            encoder (Encoder): The encoder module.
            decoder (Decoder): The decoder module.
            src_embed (InputEmbeddings): Source embedding layer.
            tgt_embed (InputEmbeddings): Target embedding layer.
            src_pos (PositionalEncoding): Source positional encoding.
            tgt_pos (PositionalEncoding): Target positional encoding.
            projection_layer (ProjectionLayer): Final projection layer.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode the source sequence.

        Args:
            src (torch.Tensor): Source sequence.
            src_mask (torch.Tensor): Source mask for attention.

        Returns:
            torch.Tensor: Encoded representation of the source.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode the target sequence.

        Args:
            encoder_output (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask for attention.
            tgt (torch.Tensor): Target sequence.
            tgt_mask (torch.Tensor): Target mask for attention.

        Returns:
            torch.Tensor: Decoded representation of the target.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the decoder output to vocabulary size.

        Args:
            x (torch.Tensor): Decoder output.

        Returns:
            torch.Tensor: Projected output.
        """
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048
) -> Transformer:
    """
    Build a Transformer model with the specified parameters.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Maximum source sequence length.
        tgt_seq_len (int): Maximum target sequence length.
        d_model (int): The model's hidden dimension.
        N (int): Number of encoder and decoder layers.
        h (int): Number of attention heads.
        dropout (float): Dropout rate.
        d_ff (int): Dimension of the feed-forward layer.

    Returns:
        Transformer: The constructed Transformer model.
    """
    # Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout
        )
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the Transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed,
        src_pos, tgt_pos, projection_layer
    )

    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer