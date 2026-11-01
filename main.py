import mlx.core as mlx
import wikipediaapi
import os

if not os.path.exists("football.txt"):
    wiki = wikipediaapi(
        user_agent="kickoff-football-bot/1.0 (aaravgarg1011@gmail.com)",
        language="en",
    )
    clubs = ["FC Barcelona", "Manchester City F.C.", "Liverpool F.C.", "Real Madrid CF"]
    text = ""