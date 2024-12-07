import torch
import torch.nn.functional as F
from model import build_transformer

# Configuration
src_vocab_size = 1000  # Vocabulary size for the source
tgt_vocab_size = 1000  # Vocabulary size for the target
src_seq_len = 10       # Source sequence length
tgt_seq_len = 10       # Target sequence length
d_model = 512          # Embedding dimension
num_layers = 6         # Number of layers in Encoder and Decoder
num_heads = 8          # Number of attention heads
dropout = 0.1          # Dropout probability
d_ff = 2048            # Dimension of feedforward layers

# Create a transformer model
transformer = build_transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    src_seq_len=src_seq_len,
    tgt_seq_len=tgt_seq_len,
    d_model=d_model,
    N=num_layers,
    h=num_heads,
    dropout=dropout,
    d_ff=d_ff
)

# Random input data
batch_size = 2
src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))  # Source input
tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))  # Target input

# Masks (1 for valid tokens, 0 for padding tokens)
src_mask = torch.ones(batch_size, 1, src_seq_len)  # Source mask
tgt_mask = torch.ones(batch_size, tgt_seq_len, tgt_seq_len).triu(diagonal=1) == 0  # Target mask for causal masking

# Forward pass
with torch.no_grad():
    encoder_output = transformer.encode(src, src_mask)
    decoder_output = transformer.decode(encoder_output, src_mask, tgt, tgt_mask)
    logits = transformer.project(decoder_output)

# Output predictions
predictions = F.softmax(logits, dim=-1)
print("Predictions shape:", predictions.shape)  # Should be (batch_size, tgt_seq_len, tgt_vocab_size)
print("Sample predictions (first element):", predictions[0])




