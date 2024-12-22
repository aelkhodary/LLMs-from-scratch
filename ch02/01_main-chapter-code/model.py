import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def process_batch(batch, token_embedding, position_embedding):
    inputs, targets = batch
    
    # Get token embeddings
    token_embeddings = token_embedding(inputs)
    
    # Get position embeddings
    positions = torch.arange(inputs.shape[1], device=inputs.device)
    pos_embeddings = position_embedding(positions)
    
    # Combine embeddings
    input_embeddings = token_embeddings + pos_embeddings
    
    return input_embeddings, targets

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_heads, n_layers):
        super().__init__()
        
        # Multi-head self-attention layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=n_heads,
                dim_feedforward=4 * embedding_dim,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Final output layer
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # Ensure input has correct shape [batch_size, sequence_length, embedding_dim]
        if len(x.shape) != 3:
            raise ValueError(f"Expected input shape [batch_size, seq_len, embedding_dim], got {x.shape}")
            
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Project to vocabulary size
        return self.output_layer(x)

# 6. Validation function
def validate(model, dataloader, criterion, token_emb, pos_emb):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            embeddings, targets = process_batch(batch, token_emb, pos_emb)
            outputs = model(embeddings)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)