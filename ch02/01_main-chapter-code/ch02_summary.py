import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from model import TransformerModel

# 1. Create the dataset class
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        # Tokenize text with allowed special tokens
        token_ids = tokenizer.encode(
            txt, 
            allowed_special={"<|endoftext|>"}
        )
        
        # Create sliding windows
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# 2. Create the data loading function
def create_dataloader(txt, batch_size=32, max_length=256, stride=128):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

# 3. Set up embeddings
def create_embeddings(vocab_size=50257, embedding_dim=256, max_length=256):
    # Token embeddings
    token_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
    
    # Position embeddings
    position_embedding = torch.nn.Embedding(max_length, embedding_dim)
    
    return token_embedding, position_embedding

# 4. Process a batch of data
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

# 5. Generate text
def generate_text(model, token_emb, pos_emb, tokenizer, prompt, max_length=100, temperature=1.0):
    model.eval()  # Set to evaluation mode
    
    # Tokenize the prompt with allowed special tokens
    input_ids = torch.tensor(
        tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    ).unsqueeze(0)  # Add batch dimension
    
    generated_tokens = input_ids[0].tolist()
    
    # Get the end token ID
    try:
        end_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    except:
        end_token_id = None
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input for the model (limit to last 256 tokens)
            curr_input = torch.tensor([generated_tokens[-256:]]).to(input_ids.device)
            
            # Get embeddings
            token_embeddings = token_emb(curr_input)
            positions = torch.arange(curr_input.shape[1], device=curr_input.device)
            pos_embeddings = pos_emb(positions).unsqueeze(0)  # Add batch dimension
            
            # Combine embeddings (ensuring correct dimensions)
            input_embeddings = token_embeddings + pos_embeddings
            
            # Forward pass
            outputs = model(input_embeddings)  # Shape: [batch_size, seq_len, vocab_size]
            
            # Get next token logits
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Append to generated sequence
            generated_tokens.append(next_token)
            
            # Stop if we generate the end token
            if end_token_id and next_token == end_token_id:
                break
    
    # Decode the generated tokens without the allowed_special parameter
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text

if __name__ == "__main__":
    # Update the path to point to the correct location
    file_path = "ch02/01_main-chapter-code/the-verdict.txt"
    
    try:
        # Load your text
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        print("Please ensure the file exists at the specified location")
        exit(1)
    
    
    # Create dataloader
    dataloader = create_dataloader(
        text,
        batch_size=32,
        max_length=256,
        stride=128
    )
    
    # Create embeddings
    token_emb, pos_emb = create_embeddings()
    
    # Initialize model, criterion, and optimizer
    model = TransformerModel(vocab_size=50257, embedding_dim=256, n_heads=8, n_layers=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for batch in dataloader:
            # Get embeddings
            embeddings, targets = process_batch(batch, token_emb, pos_emb)
            
            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Generate text
    # Save the trained model
    # Save the trained model
    torch.save(model.state_dict(), 'trained_transformer.pth')

    # Load the model for text generation
    loaded_model = TransformerModel(vocab_size=50257, embedding_dim=256, n_heads=8, n_layers=6)
    loaded_model.load_state_dict(torch.load('trained_transformer.pth'))
    loaded_model.eval()  # Set the model to evaluation mode

    # Generate text
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Did She sent for paint him when he was dead."
    
    print("\nGenerating text from prompt:", prompt)
    generated_text = generate_text(
        model=loaded_model,
        token_emb=token_emb,
        pos_emb=pos_emb,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=100,
        temperature=0.8  # Lower temperature (e.g., 0.7) for more focused text
                        # Higher temperature (e.g., 1.2) for more creative text
    )
    print("\nGenerated text:")
    print(generated_text)
