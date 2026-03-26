#!/usr/bin/env python3
"""
Fetch Formula 1 constructor standings for seasons 2003-2025
from the Jolpica F1 API and save as JSON.
"""

import json
import time
import urllib.request
import urllib.error


BASE_URL = "https://api.jolpi.ca/ergast/f1/{year}/constructorStandings.json"
OUTPUT_PATH = "/Users/corneliuswiehl/Documents/projects/startup/projects/luck_vs_skill/visualization/f1_data.json"

START_YEAR = 2003
END_YEAR = 2025


def fetch_season(year):
    """Fetch constructor standings for a single season."""
    url = BASE_URL.format(year=year)
    req = urllib.request.Request(url, headers={"User-Agent": "F1DataFetcher/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data


def parse_season(year, raw):
    """Extract team names, points, wins, position from raw API response."""
    standings_table = raw["MRData"]["StandingsTable"]
    if not standings_table.get("StandingsLists"):
        return None

    standings_list = standings_table["StandingsLists"][0]
    season = int(standings_list["season"])
    round_number = int(standings_list["round"])  # number of races completed

    constructor_standings = standings_list["ConstructorStandings"]

    teams = []
    points = []
    wins = []
    positions = []

    for i, entry in enumerate(constructor_standings):
        teams.append(entry["Constructor"]["name"])
        points.append(float(entry["points"]))
        wins.append(int(entry["wins"]))
        # Some entries (e.g. 2007 last team) lack 'position' or have non-numeric positionText
        pos_raw = entry.get("position") or entry.get("positionText") or str(i + 1)
        try:
            positions.append(int(pos_raw))
        except ValueError:
            # Non-numeric (e.g. "E" for excluded) - use 1-based index
            positions.append(i + 1)

    n_teams = len(teams)

    # Compute observed variance of the points distribution
    if n_teams > 0:
        mean_pts = sum(points) / n_teams
        observed_variance = sum((p - mean_pts) ** 2 for p in points) / n_teams
    else:
        observed_variance = 0.0

    return {
        "year": season,
        "teams": teams,
        "points": points,
        "wins": wins,
        "positions": positions,
        "n_teams": n_teams,
        "n_races": round_number,
        "observed_variance": round(observed_variance, 2),
    }


def main():
    seasons = []
    failed = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Fetching {year}...", end=" ", flush=True)
        success = False
        for attempt in range(3):
            try:
                raw = fetch_season(year)
                parsed = parse_season(year, raw)
                if parsed is None:
                    print(f"NO DATA (empty standings list)")
                    failed.append(year)
                else:
                    seasons.append(parsed)
                    print(
                        f"OK  ->  {parsed['n_teams']} teams, "
                        f"{parsed['n_races']} races, "
                        f"variance={parsed['observed_variance']}"
                    )
                success = True
                break
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, Exception) as e:
                if attempt < 2:
                    print(f"retry({attempt+1})...", end=" ", flush=True)
                    time.sleep(1)
                else:
                    print(f"FAILED after 3 attempts ({e})")
                    failed.append(year)

        # polite delay between requests
        if year < END_YEAR:
            time.sleep(0.5)

    # Build summary
    if seasons:
        avg_observed_variance = round(
            sum(s["observed_variance"] for s in seasons) / len(seasons), 2
        )
        avg_teams = round(sum(s["n_teams"] for s in seasons) / len(seasons), 2)
        avg_races = round(sum(s["n_races"] for s in seasons) / len(seasons), 2)
    else:
        avg_observed_variance = 0.0
        avg_teams = 0.0
        avg_races = 0.0

    result = {
        "seasons": seasons,
        "summary": {
            "avg_observed_variance": avg_observed_variance,
            "avg_teams": avg_teams,
            "n_seasons": len(seasons),
            "avg_races": avg_races,
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone. {len(seasons)} seasons saved to {OUTPUT_PATH}")
    if failed:
        print(f"Failed/empty seasons: {failed}")

    # Print summary
    print(f"\n--- Summary ---")
    print(f"Seasons:              {result['summary']['n_seasons']}")
    print(f"Avg teams per season: {result['summary']['avg_teams']}")
    print(f"Avg races per season: {result['summary']['avg_races']}")
    print(f"Avg observed variance:{result['summary']['avg_observed_variance']}")


if __name__ == "__main__":
    main()
