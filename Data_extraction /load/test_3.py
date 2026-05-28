from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import time

# Trouver l'ID de Harden
harden = players.find_players_by_full_name("James Harden")[0]
print(f"Harden ID : {harden['id']}")

# Game log individuel
log = playergamelog.PlayerGameLog(
    player_id=harden['id'],
    season="2024-25",
    season_type_all_star="Regular Season"
)

df = log.get_data_frames()[0]
print(f"Matchs : {len(df)}")
print(df[["Game_ID", "GAME_DATE", "MATCHUP", "MIN", "PTS"]].head(10).to_string())