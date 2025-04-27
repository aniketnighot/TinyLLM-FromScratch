import torch
from torch.nn import functional as F
from biagram_model.model import BiogramLLMModel
from biagram_model.utils import get_vocab, get_numerical_tokens, decode_tokens

# Set hyperparameters
context_length = 128
batch_size = 32
num_embed = 128
num_heads = 8
num_layers = 6
dropout = 0.2
max_iterations = 5000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Read data
with open('hindi_song_lyrics.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab, vocab_dict = get_vocab(text)
vocab_size = len(vocab)

# Prepare dataset
data = torch.tensor(get_numerical_tokens(vocab_dict, text), dtype=torch.long)
train_data = data[:int(0.9*len(data))]
val_data = data[int(0.9*len(data)):]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(0, len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in idx])
    y = torch.stack([data[i+1:i+context_length+1] for i in idx])
    return x.to(device), y.to(device)

# Model
model = BiogramLLMModel(vocab_size, context_length, num_embed, num_heads, num_layers, dropout)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training
for iter in range(max_iterations):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 500 == 0:
        print(f"Step {iter}: Loss = {loss.item():.4f}")

# Generate sample text
context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_tokens = model.generate(context, max_new_tokens=500, context_length=context_length)
print(decode_tokens(vocab, generated_tokens[0].tolist()))

# Save model
torch.save(model.state_dict(), 'biogram_model.pth')
