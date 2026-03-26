#!/usr/bin/env python3
"""
Compute luck vs skill metrics for eSports match data (CS:GO, Dota2, LoL).

Method: Compare observed variance of team win rates to the variance expected
under pure luck (binomial with p=0.5). The ratio luck_variance / observed_variance
gives the fraction of outcome variation attributable to luck.
"""

import pandas as pd
import numpy as np
import json
import os

# =============================================================================
# 1. Load and inspect each CSV
# =============================================================================
DATA_DIR = "/Users/corneliuswiehl/Documents/projects/startup/projects/luck_vs_skill/data/esports_data"
OUT_PATH = "/Users/corneliuswiehl/Documents/projects/startup/projects/luck_vs_skill/visualization/esports_data.json"

files = {
    "CSGO":  os.path.join(DATA_DIR, "csgo_matches.csv"),
    "Dota2": os.path.join(DATA_DIR, "dota2_matches.csv"),
    "LoL":   os.path.join(DATA_DIR, "lol_matches.csv"),
}

datasets = {}
for game, path in files.items():
    print(f"\n{'='*70}")
    print(f"Loading {game} from {path}")
    print(f"{'='*70}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"\nHead:\n{df.head()}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nUnique 'win' values: {df['win'].unique()}")
    print(f"Unique teams: {df['team'].nunique()}")
    print(f"Unique tournaments: {df['tournament'].nunique()}")

    # Parse the date column — format is like "Tuesday, 13 December 2022"
    df['match_date'] = pd.to_datetime(df['match_date'], format="%A, %d %B %Y")
    df['year'] = df['match_date'].dt.year
    print(f"Year range: {df['year'].min()} – {df['year'].max()}")
    print(f"Rows per year:\n{df['year'].value_counts().sort_index()}")

    # Convert win column to binary
    df['is_win'] = (df['win'] == 'win').astype(int)

    datasets[game] = df

# =============================================================================
# 2-3. Compute luck vs skill for each game, per year
# =============================================================================
MIN_TEAMS = 8
MIN_MATCHES_PER_TEAM = 20

results = {}

for game, df in datasets.items():
    print(f"\n{'='*70}")
    print(f"Computing luck vs skill for {game}")
    print(f"{'='*70}")

    year_records = []

    for year, ydf in df.groupby('year'):
        # --- 2b. Find teams with enough matches this year ---
        team_stats = ydf.groupby('team').agg(
            n_matches=('is_win', 'count'),
            n_wins=('is_win', 'sum')
        ).reset_index()

        # Keep only teams with at least MIN_MATCHES_PER_TEAM matches
        team_stats = team_stats[team_stats['n_matches'] >= MIN_MATCHES_PER_TEAM].copy()

        n_teams = len(team_stats)
        if n_teams < MIN_TEAMS:
            print(f"  {year}: only {n_teams} qualifying teams (need {MIN_TEAMS}) — skipping")
            continue

        # --- 2c. Win rate per team ---
        team_stats['win_rate'] = team_stats['n_wins'] / team_stats['n_matches']

        # --- 2d. Observed variance of win rates ---
        obs_variance = team_stats['win_rate'].var(ddof=0)  # population variance

        # --- 2e. Luck variance under binomial assumption ---
        # For each team i with n_i matches, luck variance of its win rate = 0.25 / n_i.
        # The average luck variance across teams:
        avg_n = team_stats['n_matches'].mean()
        luck_variance = (0.25 / team_stats['n_matches']).mean()

        # --- 2f. Luck contribution ---
        if obs_variance > 0:
            luck_pct = luck_variance / obs_variance
        else:
            luck_pct = np.nan

        year_records.append({
            "year": int(year),
            "n_teams": int(n_teams),
            "avg_matches": round(float(avg_n), 1),
            "obs_variance": round(float(obs_variance), 6),
            "luck_variance": round(float(luck_variance), 6),
            "luck_pct": round(float(luck_pct), 4),
        })

        print(f"  {year}: {n_teams} teams, avg {avg_n:.0f} matches, "
              f"obs_var={obs_variance:.4f}, luck_var={luck_variance:.4f}, "
              f"luck%={luck_pct:.2%}")

    # Sort by year
    year_records.sort(key=lambda r: r['year'])

    # Summary
    if year_records:
        avg_luck = np.mean([r['luck_pct'] for r in year_records])
    else:
        avg_luck = np.nan

    results[game] = {
        "years": year_records,
        "avg_luck": round(float(avg_luck), 4),
        "n_seasons": len(year_records),
    }

    print(f"\n  >>> {game} summary: {len(year_records)} qualifying seasons, "
          f"average luck contribution = {avg_luck:.2%}")

# =============================================================================
# 4. Save to JSON
# =============================================================================
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print(f"Results saved to {OUT_PATH}")
print(f"{'='*70}")

# Final overview
print("\n=== FINAL OVERVIEW ===")
for game, data in results.items():
    print(f"  {game}: {data['n_seasons']} seasons, avg luck = {data['avg_luck']:.2%}")
