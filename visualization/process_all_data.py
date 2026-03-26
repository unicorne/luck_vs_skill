"""
Comprehensive data processing script for Luck vs Skill visualization.

Reads:
  - ESPN standings data (2022-2025)
  - eSports luck metrics (CS:GO, Dota2, LoL)
  - Original parquet results (2003-2021)

Computes:
  - Monte Carlo luck simulations for ESPN leagues (new seasons)
  - Merges old parquet data with new ESPN data

Outputs:
  - all_data.json: unified JSON for the visualization
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = BASE_DIR / "visualization"
DATA_DIR = BASE_DIR / "data" / "results"

N_SIMULATIONS = 1000  # Monte Carlo simulations for robustness

# ============================================================
# League configuration
# ============================================================

LEAGUE_CONFIG = {
    "NBA": {
        "name": "NBA",
        "sport": "Basketball",
        "country": "North America",
        "icon": "\U0001F3C0",
        "color": "#f97316",
        "espn_key": "NBA",
        "parquet_key": "NBA",
        "points_per_win": 2,
        "points_per_tie": 0,
        "points_per_loss": 0,
        "tie_prob": 0.0,
        "full_season_games": 82,
    },
    "NFL": {
        "name": "NFL",
        "sport": "Football",
        "country": "North America",
        "icon": "\U0001F3C8",
        "color": "#16a34a",
        "espn_key": "NFL",
        "parquet_key": "NFL",
        "points_per_win": 1,
        "points_per_tie": 0.5,
        "points_per_loss": 0,
        "tie_prob": 0.015,
        "full_season_games": 17,
    },
    "NHL": {
        "name": "NHL",
        "sport": "Hockey",
        "country": "North America",
        "icon": "\U0001F3D2",
        "color": "#2563eb",
        "espn_key": "NHL",
        "parquet_key": "NHL",
        "points_per_win": 2,
        "points_per_tie": 1,  # OT loss
        "points_per_loss": 0,
        "tie_prob": 0.10,  # OT loss probability
        "full_season_games": 82,
    },
    "MLB": {
        "name": "MLB",
        "sport": "Baseball",
        "country": "North America",
        "icon": "\u26BE",
        "color": "#dc2626",
        "espn_key": "MLB",
        "parquet_key": "MLB",
        "points_per_win": 1,
        "points_per_tie": 0,
        "points_per_loss": 0,
        "tie_prob": 0.0,
        "full_season_games": 162,
    },
    "PML": {
        "name": "Premier League",
        "sport": "Soccer",
        "country": "England",
        "icon": "\u26BD",
        "color": "#6d28d9",
        "espn_key": "Premier League",
        "parquet_key": "PML",
        "points_per_win": 3,
        "points_per_tie": 1,
        "points_per_loss": 0,
        "tie_prob": 0.25,
        "full_season_games": 38,
    },
    "LaLiga": {
        "name": "La Liga",
        "sport": "Soccer",
        "country": "Spain",
        "icon": "\u26BD",
        "color": "#ea580c",
        "espn_key": "La Liga",
        "parquet_key": "LaLiga",
        "points_per_win": 3,
        "points_per_tie": 1,
        "points_per_loss": 0,
        "tie_prob": 0.25,
        "full_season_games": 38,
    },
    "Bundesliga": {
        "name": "Bundesliga",
        "sport": "Soccer",
        "country": "Germany",
        "icon": "\u26BD",
        "color": "#b91c1c",
        "espn_key": "Bundesliga",
        "parquet_key": "Bundesliga",
        "points_per_win": 3,
        "points_per_tie": 1,
        "points_per_loss": 0,
        "tie_prob": 0.25,
        "full_season_games": 34,
    },
    "Ligue1": {
        "name": "Ligue 1",
        "sport": "Soccer",
        "country": "France",
        "icon": "\u26BD",
        "color": "#0369a1",
        "espn_key": "Ligue 1",
        "parquet_key": "Ligue1",
        "points_per_win": 3,
        "points_per_tie": 1,
        "points_per_loss": 0,
        "tie_prob": 0.25,
        "full_season_games": 38,
    },
    "SerieA": {
        "name": "Serie A",
        "sport": "Soccer",
        "country": "Italy",
        "icon": "\u26BD",
        "color": "#059669",
        "espn_key": "Serie A",
        "parquet_key": "SerieA",
        "points_per_win": 3,
        "points_per_tie": 1,
        "points_per_loss": 0,
        "tie_prob": 0.25,
        "full_season_games": 38,
    },
}


# ============================================================
# Monte Carlo simulation for traditional leagues
# ============================================================

def simulate_league_season(n_teams, n_games, points_per_win, points_per_loss,
                           points_per_tie, tie_prob, n_sims=N_SIMULATIONS):
    """
    Simulate n_sims random seasons and return the average variance of point totals.

    Each team plays n_games independently. For each game:
      - With probability tie_prob, it's a draw/OT loss (team gets points_per_tie)
      - Otherwise, 50/50 win or loss
    """
    all_variances = []
    for _ in range(n_sims):
        # Vectorized: (n_teams, n_games) random outcomes
        rand = np.random.random((n_teams, n_games))
        # Determine outcome: tie if rand < tie_prob, else win/loss with 50%
        # For non-tie games, win if the remaining random is < 0.5 of the non-tie range
        is_tie = rand < tie_prob
        # For remaining games: win with 50% probability
        is_win = (~is_tie) & (rand < tie_prob + (1 - tie_prob) * 0.5)
        is_loss = (~is_tie) & (~is_win)

        points = (is_win.astype(float) * points_per_win +
                  is_tie.astype(float) * points_per_tie +
                  is_loss.astype(float) * points_per_loss)
        season_totals = points.sum(axis=1)
        # Population variance
        var = np.var(season_totals)
        all_variances.append(var)

    return np.mean(all_variances)



# ============================================================
# Helper functions
# ============================================================

def clean_team_name(name):
    """Remove ranking prefixes and codes from team names."""
    import re
    # Remove patterns like "1 CHI ", "z -- PHX ", "* -- NYY ", "y -- SA ", "x -- SF "
    # Pattern: optional symbols, optional " -- ", optional 2-4 letter code, space
    cleaned = re.sub(r'^[\d*xyzw\-\s]+--\s+[A-Z]{2,4}\s+', '', name)
    # Also handle "1 CHI " style (MLS, soccer)
    cleaned = re.sub(r'^\d+\s+[A-Z]{2,5}\s+', '', cleaned)
    return cleaned.strip()


def extract_parquet_yearly_data(league_key, config):
    """Extract year-by-year data from a parquet file."""
    parquet_path = DATA_DIR / f"{config['parquet_key']}.parquet"
    if not parquet_path.exists():
        print(f"  WARNING: {parquet_path} not found")
        return []

    df = pd.read_parquet(parquet_path)
    yearly = []
    for year in df.index:
        row = df.loc[year]
        teams = row['Teams']
        points = row['Points']
        n_teams = row['#Teams']
        obs_var = row['Variance_observed']
        luck_var = row['Luck_variance']

        if isinstance(teams, np.ndarray):
            teams = teams.tolist()
        if isinstance(points, np.ndarray):
            points = [float(p) for p in points]

        # Find top team
        if len(teams) > 0:
            top_team = clean_team_name(str(teams[0]))
        else:
            top_team = "Unknown"

        top_points = float(max(points)) if len(points) > 0 else 0
        bottom_points = float(min(points)) if len(points) > 0 else 0

        luck_pct = luck_var / obs_var if obs_var > 0 else 0

        yearly.append({
            "year": int(year),
            "obs_var": round(float(obs_var), 2),
            "luck_var": round(float(luck_var), 2),
            "luck_pct": round(float(luck_pct), 4),
            "n_teams": int(n_teams),
            "top_team": top_team,
            "top_points": round(top_points, 1),
            "bottom_points": round(bottom_points, 1),
        })

    return yearly


# ============================================================
# Main processing
# ============================================================

def main():
    print("=" * 60)
    print("LUCK vs SKILL - Data Processing Pipeline")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Load all input data
    # ----------------------------------------------------------
    print("\n[1/4] Loading input data...")

    with open(VIZ_DIR / "espn_data.json") as f:
        espn_data = json.load(f)
    print(f"  ESPN data: {len(espn_data)} leagues")

    with open(VIZ_DIR / "esports_data.json") as f:
        esports_data = json.load(f)
    print(f"  eSports data: {list(esports_data.keys())}")

    # Load summary results for reference
    results_df = pd.read_parquet(DATA_DIR / "results_df.parquet")
    print(f"  Results summary: {len(results_df)} sports")

    # ----------------------------------------------------------
    # 2. Process each traditional league
    # ----------------------------------------------------------
    print("\n[2/4] Processing traditional leagues...")

    leagues_output = {}

    for league_key, config in LEAGUE_CONFIG.items():
        print(f"\n  --- {config['name']} ---")

        # Get parquet data (historical)
        yearly_data = extract_parquet_yearly_data(league_key, config)
        existing_years = {d['year'] for d in yearly_data}
        print(f"  Parquet years: {sorted(existing_years)}")

        # Get ESPN data (new seasons 2022-2025)
        espn_key = config['espn_key']
        if espn_key in espn_data:
            espn_seasons = espn_data[espn_key]
            new_years = sorted(espn_seasons.keys())
            print(f"  ESPN years: {new_years}")

            for year_str in new_years:
                year = int(year_str)
                if year in existing_years:
                    print(f"    Year {year}: already in parquet, skipping ESPN")
                    continue

                season = espn_seasons[year_str]
                n_teams = len(season['teams'])
                n_games = season['games']
                points = season['points']

                # Check if season is complete
                full_games = config.get('full_season_games', n_games)
                is_partial = n_games < full_games * 0.95  # <95% of full season

                # Compute observed variance (population)
                points_arr = np.array(points, dtype=float)
                obs_var = float(np.var(points_arr))

                # Simulate luck variance
                luck_var = simulate_league_season(
                    n_teams=n_teams,
                    n_games=n_games,
                    points_per_win=config['points_per_win'],
                    points_per_loss=config['points_per_loss'],
                    points_per_tie=config['points_per_tie'],
                    tie_prob=config['tie_prob'],
                )

                luck_pct = luck_var / obs_var if obs_var > 0 else 0

                top_team = season['teams'][0] if season['teams'] else "Unknown"
                top_points = float(max(points)) if points else 0
                bottom_points = float(min(points)) if points else 0

                entry = {
                    "year": year,
                    "obs_var": round(obs_var, 2),
                    "luck_var": round(luck_var, 2),
                    "luck_pct": round(luck_pct, 4),
                    "n_teams": n_teams,
                    "top_team": top_team,
                    "top_points": round(top_points, 1),
                    "bottom_points": round(bottom_points, 1),
                }
                if is_partial:
                    entry["partial_season"] = True
                    entry["games_played"] = n_games
                    entry["full_season_games"] = full_games
                yearly_data.append(entry)
                partial_str = f" [PARTIAL {n_games}/{full_games}]" if is_partial else ""
                print(f"    Year {year}: obs_var={obs_var:.2f}, luck_var={luck_var:.2f}, luck_pct={luck_pct:.4f}{partial_str}")
        else:
            print(f"  No ESPN data found for {espn_key}")

        # Sort by year
        yearly_data.sort(key=lambda x: x['year'])

        # Compute overall averages
        if yearly_data:
            avg_luck = np.mean([d['luck_pct'] for d in yearly_data])
            avg_teams = np.mean([d['n_teams'] for d in yearly_data])
        else:
            avg_luck = 0
            avg_teams = 0

        # Get avg games from config or results_df
        sport_row = results_df[results_df['Sport'] == config['parquet_key']]
        if not sport_row.empty:
            avg_games = float(sport_row['#Games (avg)'].iloc[0])
        else:
            avg_games = config.get('default_games', 0)

        leagues_output[league_key] = {
            "name": config['name'],
            "sport": config['sport'],
            "country": config['country'],
            "icon": config['icon'],
            "color": config['color'],
            "avg_teams": round(float(avg_teams), 1),
            "avg_games": round(avg_games, 1),
            "total_seasons": len(yearly_data),
            "luck_contribution": round(float(avg_luck), 4),
            "yearly_data": yearly_data,
        }

        print(f"  Total seasons: {len(yearly_data)}, Avg luck: {avg_luck:.4f}")

    # ----------------------------------------------------------
    # 3. Process eSports
    # ----------------------------------------------------------
    print("\n[3/4] Processing eSports data...")

    esports_output = {}

    esports_config = {
        "CSGO": {
            "name": "CS:GO",
            "sport": "eSports",
            "country": "International",
            "icon": "\U0001F3AE",
            "color": "#f59e0b",
        },
        "Dota2": {
            "name": "Dota 2",
            "sport": "eSports",
            "country": "International",
            "icon": "\U0001F3AE",
            "color": "#ef4444",
        },
        "LoL": {
            "name": "League of Legends",
            "sport": "eSports",
            "country": "International",
            "icon": "\U0001F3AE",
            "color": "#3b82f6",
        },
    }

    for esport_key, es_config in esports_config.items():
        if esport_key not in esports_data:
            print(f"  WARNING: {esport_key} not found in esports data")
            continue

        es_data = esports_data[esport_key]
        yearly = []
        for year_data in es_data['years']:
            yearly.append({
                "year": year_data['year'],
                "n_teams": year_data['n_teams'],
                "avg_matches": year_data['avg_matches'],
                "obs_variance": round(year_data['obs_variance'], 6),
                "luck_variance": round(year_data['luck_variance'], 6),
                "luck_pct": round(year_data['luck_pct'], 4),
            })

        avg_luck = es_data['avg_luck']
        avg_teams = np.mean([y['n_teams'] for y in yearly])
        avg_matches = np.mean([y['avg_matches'] for y in yearly])

        esports_output[esport_key] = {
            "name": es_config['name'],
            "sport": es_config['sport'],
            "country": es_config['country'],
            "icon": es_config['icon'],
            "color": es_config['color'],
            "avg_teams": round(float(avg_teams), 1),
            "avg_matches": round(float(avg_matches), 1),
            "total_seasons": es_data['n_seasons'],
            "luck_contribution": round(float(avg_luck), 4),
            "yearly_data": yearly,
        }

        print(f"  {es_config['name']}: {es_data['n_seasons']} seasons, avg luck: {avg_luck:.4f}")

    # ----------------------------------------------------------
    # 4. Assemble and write output
    # ----------------------------------------------------------
    print("\n[4/4] Assembling unified output...")

    output = {
        "leagues": leagues_output,
        "new_sports": {
            **esports_output,
        },
        "metadata": {
            "generated": "2026-03-25",
            "n_simulations": N_SIMULATIONS,
            "description": "Luck vs Skill metrics across sports leagues and eSports",
            "sources": {
                "traditional_leagues": "ESPN standings (2022-2025) + historical parquet data",
                "esports": "Pre-computed win-rate variance analysis",
            },
        },
    }

    output_path = VIZ_DIR / "all_data.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Output written to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY - Luck Contributions (lower = more skill-driven)")
    print("=" * 60)

    all_items = []
    for key, league in leagues_output.items():
        all_items.append((league['name'], league['luck_contribution'], league['total_seasons'], "league"))
    for key, es in esports_output.items():
        all_items.append((es['name'], es['luck_contribution'], es['total_seasons'], "new"))

    all_items.sort(key=lambda x: x[1])

    print(f"\n{'Sport':<25} {'Luck %':>10} {'Seasons':>10} {'Category':>12}")
    print("-" * 60)
    for name, luck, seasons, cat in all_items:
        print(f"{name:<25} {luck*100:>9.2f}% {seasons:>10} {cat:>12}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
