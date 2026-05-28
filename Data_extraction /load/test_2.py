from nba_api.stats.endpoints import playergamelogs
import time

# Récupérer avec SeasonType explicite et vérifier la granularité
logs = playergamelogs.PlayerGameLogs(
    season_nullable="2024-25",
    season_type_nullable="Regular Season",
    per_mode_simple_nullable="Totals"  # minutes totales par match
)

df = logs.get_data_frames()[0]

# Filtrer sur un seul joueur pour vérifier
harden = df[df["PLAYER_NAME"] == "James Harden"].sort_values("GAME_DATE")
print(f"Matchs de Harden : {len(harden)}")
print(harden[["PLAYER_NAME", "GAME_DATE", "MIN", "MATCHUP"]].head(10).to_string())