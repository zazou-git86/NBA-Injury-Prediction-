from nba_api.stats.endpoints import playergamelogs
import time

# Récupérer tous les game logs de la saison 2024-25
logs = playergamelogs.PlayerGameLogs(
    season_nullable="2024-25",
    season_type_nullable="Regular Season"
)

df = logs.get_data_frames()[0]
print(f"Lignes : {len(df):,}")
print(f"Colonnes : {df.columns.tolist()}")
print(df[['PLAYER_NAME', 'GAME_DATE', 'MIN', 'MATCHUP']].head(10))