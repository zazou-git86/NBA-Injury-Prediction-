from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import time

# Tester la structure sur 3 joueurs avant de lancer le gros script
test_players = ["James Harden", "LeBron James", "Stephen Curry"]

results = []
for name in test_players:
    player = players.find_players_by_full_name(name)[0]
    log = playergamelog.PlayerGameLog(
        player_id=player['id'],
        season="2024-25",
        season_type_all_star="Regular Season"
    )
    df = log.get_data_frames()[0]
    df["PLAYER_ID"] = player['id']
    df["PLAYER_NAME"] = name
    results.append(df)
    print(f"{name} : {len(df)} matchs | mins moy : {df['MIN'].mean():.1f}")
    time.sleep(1)  # délai anti-ban

full = pd.concat(results, ignore_index=True)
print(f"\nColonnes disponibles : {full.columns.tolist()}")
print(f"\nAperçu :\n{full[['PLAYER_NAME','GAME_DATE','MATCHUP','MIN','PTS','REB','AST']].head(9).to_string()}")