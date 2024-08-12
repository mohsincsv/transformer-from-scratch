# Transformers from Scratch

I've implemented the Transformer architecture for machine translation, as described in "Attention Is All You Need" by Vaswani et al. (2017), using PyTorch without relying on high-level transformer libraries.

## Project Overview

In this project, I:

- Built the core Transformer architecture, including multi-head attention, positional encoding, and feed-forward networks.
- Designed a `BilingualDataset` class for efficient processing of translation pairs.
- Implemented a training pipeline with custom learning rate scheduling and gradient clipping.
- Created a translation function using greedy decoding for inference.
- Utilized Hugging Face's tokenizers for flexible text tokenization.
- Integrated TensorBoard for visualizing training metrics.

## Main Components

The main components of the architecture include:

- **Encoder and Decoder Stacks**: Customizable number of layers to tailor the model complexity.
- **Multi-Head Attention Mechanism**: Captures complex relationships in the input through multiple attention heads.
- **Position-wise Feed-Forward Networks**: Provides additional transformation of features after the attention mechanism.
- **Layer Normalization and Residual Connections**: Ensures stable training by normalizing activations and adding skip connections.
- **Positional Encoding**: Injects sequence order information into the model to account for the position of words in the sentence.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Happy Building 🚀😊
