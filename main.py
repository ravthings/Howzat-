import mlx.core as mlx
import mlx.nn as nn
import mlx.optimizers as optim
import wikipediaapi
import os
import re

if not os.path.exists("football.txt"):
    wiki = wikipediaapi.Wikipedia(
        user_agent="kickoff-football-bot/1.0 (aaravgarg1011@gmail.com)",
        language="en",
    )
    clubs = [
    "FC Barcelona", "Manchester City F.C.", "Liverpool F.C.", "Real Madrid CF",
    "Arsenal F.C.", "Chelsea F.C.", "Manchester United F.C.", "Juventus F.C.",
    "AC Milan", "Paris Saint-Germain F.C.", "Atlético Madrid", "Tottenham Hotspur F.C.",
]
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

with open("football.txt", "r", encoding="utf-8") as F:
    text = F.read()
def tokenize(s):
    return re.findall(r"\w+|[^\w\s]", s.lower())

words = tokenize(text)

vocab = sorted(set(words))
stoi = {w: idx for idx, w in enumerate(vocab)}
itos = {idx: w for w, idx in stoi.items()}

def encode(s):
    return [stoi[w] for w in tokenize(s)]
def decode(ids):
    return " ".join(itos[i] for i in ids)

ids = encode(text)
chunk = 256
train_ids, val_ids = [], []
for i in range(0, len(ids), chunk):
    if (i // chunk) % 10 == 9:
        val_ids.extend(ids[i : i + chunk])
    else:
        train_ids.extend(ids[i : i + chunk])

train_data = mlx.array(train_ids, dtype=mlx.int64)
val_data = mlx.array(val_ids, dtype=mlx.int64)

# Hyperparameters
batch_size = 32
block_size = 64
vocab_size = len(vocab)
n_embd = 64
n_head = 4

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = mlx.random.randint(0, len(d) - block_size, (batch_size,)).tolist()
    x = mlx.stack([d[i : i + block_size] for i in ix])
    y = mlx.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x, y

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def __call__(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, mask=mask)   # gather info from earlier words
        x = x + self.mlp(self.ln2(x))           # think about it
        return x

class HowzatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.block = Block()                      # replaces the bare attention
        self.ln_f = nn.LayerNorm(n_embd)          # final tidy-up
        self.head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(mlx.arange(T))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        x = self.block(x, mask)
        return self.head(self.ln_f(x))

model = HowzatModel()

def loss_fn(model, x, y):
    logits = model(x)
    B, T, C = logits.shape
    return nn.losses.cross_entropy(logits.reshape(B * T, C), y.reshape(B * T)).mean()

xb, yb = get_batch("train")
print(loss_fn(model, xb, yb).item())

def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])[:, -1, :]
        next_word = mlx.random.categorical(logits, num_samples=1)
        idx = mlx.concatenate([idx, next_word.astype(idx.dtype)], axis=1)
    return idx

training = input("Do you want to train the model? y/n ")

if training == "y":
    lr = float(input("What do you want the Learning Rate to be? "))
    optimizer = optim.AdamW(learning_rate=lr)

    if os.path.exists("kickoff.safetensors"):
        model.load_weights("kickoff.safetensors")
        print("Resuming from saved model")
    else:
        print("No saved model, training from scratch")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    epochs = input("How many epochs do you want to train for? ")
    epoche = int(epochs)
    epochs = epoche
    for epoch in range(epochs):
        xb, yb = get_batch("train")
        loss, grads = loss_and_grad(model, xb, yb)
        optimizer.update(model, grads)
        mlx.eval(model.parameters(), optimizer.state)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    model.save_weights("kickoff.safetensors")

    start = mlx.array([encode("Liverpool")])
    out = generate(model, start, max_new_tokens=50)
    print(decode(out[0].tolist()))
elif training == "n":
    if not os.path.exists("kickoff.safetensors"):
        print("No saved model found. Train it first.")
    else:
        model.load_weights("kickoff.safetensors")
        model.eval()

        xb, yb = get_batch("train")
        print(f"Train Loss = {loss_fn(model, xb, yb).item()}")

        xb, yb = get_batch("val")
        print(f"Val Loss = {loss_fn(model, xb, yb).item()}")

        start = mlx.array([encode("Liverpool")])
        out = generate(model, start, max_new_tokens=50)
        print(decode(out[0].tolist()))
else:
    print("Sorry, that is not a valid input.")