from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import players as nba_players
import pandas as pd
import time

# Test sur 3 joueurs
test_ids = [2544, 201935, 201939]  # LeBron, Harden, Curry

for pid in test_ids:
    info = commonplayerinfo.CommonPlayerInfo(player_id=pid)
    df   = info.get_data_frames()[0]
    print(f"\nID {pid} :")
    print(f"  Nom      : {df['DISPLAY_FIRST_LAST'].values[0]}")
    print(f"  Poste    : {df['POSITION'].values[0]}")
    print(f"  Naissance: {df['BIRTHDATE'].values[0]}")
    print(f"  Draft    : {df['DRAFT_YEAR'].values[0]}")
    print(f"  Taille   : {df['HEIGHT'].values[0]}")
    print(f"  Poids    : {df['WEIGHT'].values[0]}")
    time.sleep(1)