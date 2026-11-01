import mlx.core as mlx
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
