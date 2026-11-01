import mlx.core as mlx
import mlx.nn as nn
import wikipediaapi
import os

if not os.path.exists("football.txt"):
    wiki = wikipediaapi.Wikipedia(
        user_agent="kickoff-football-bot/1.0 (aaravgarg1011@gmail.com)",
        language="en",
    )
    clubs = ["FC Barcelona", "Manchester City F.C.", "Liverpool F.C.", "Real Madrid CF"]
    text = ""

    for club in clubs:
        page = wiki.page(club)
        if page.exists():
            text += page.text + "\n\n"
            print(f"Got {club}: {len(page.text)} chars")
        else:
            print(f"NOT FOUND: {club}  (check the title)")
    with open("football.txt", "w", encoding="utf-8") as f:
        f.write(text)

with open('football.txt', 'r') as F:
    text = F.read()
words = text.split()

vocab = sorted(set(words))
stoi = {w: idx for idx, w in enumerate(vocab)}
itos = {idx: w for w, idx in stoi.items()}

def encode(s):
    return [stoi[w] for w in s.split()]
def decode(ids):
    return " ".join(itos[i] for i in ids)

data = mlx.array(encode(text), dtype=mlx.int64)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#Hyperparamters
batch_size = 32
block_size = 64
vocab_size = len(vocab)

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = mlx.randint(len(d) - block_size, (batch_size,))
    x = mlx.stack([d[i: i + block_size]for i in ix])
    y = mlx.stack([d[i+1: i+block_size+1]for i in ix])
    return x, y
xb, yb = get_batch('train')

class KickOffModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, idx):
        logits = self.embedding(idx)
        return logits

model = KickOffModel()
logits = model(xb)

loss_fn = nn.CrossEntropyLoss()
optimizer = mlx.optimizers.AdamW(params=model.parameters(), lr=1e-2)
