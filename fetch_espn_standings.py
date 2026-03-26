#!/usr/bin/env python3
"""
Fetch historical sports standings data from the ESPN hidden API
for multiple leagues and seasons (2022-2025).
"""

import urllib.request
import json
import time
import sys

# League configurations
LEAGUES = {
    "NBA": {
        "url": "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings?season={year}",
        "points_system": "nba",  # 2 per win
    },
    "NFL": {
        "url": "https://site.api.espn.com/apis/v2/sports/football/nfl/standings?season={year}",
        "points_system": "nfl",  # 1 per win, 0.5 per tie
    },
    "NHL": {
        "url": "https://site.api.espn.com/apis/v2/sports/hockey/nhl/standings?season={year}",
        "points_system": "nhl",  # 2 per win, 1 per OT loss
    },
    "MLB": {
        "url": "https://site.api.espn.com/apis/v2/sports/baseball/mlb/standings?season={year}",
        "points_system": "mlb",  # 1 per win
    },
    "MLS": {
        "url": "https://site.api.espn.com/apis/v2/sports/soccer/usa.1/standings?season={year}",
        "points_system": "soccer",  # 3 per win, 1 per draw
    },
    "Premier League": {
        "url": "https://site.api.espn.com/apis/v2/sports/soccer/eng.1/standings?season={year}",
        "points_system": "soccer",
    },
    "La Liga": {
        "url": "https://site.api.espn.com/apis/v2/sports/soccer/esp.1/standings?season={year}",
        "points_system": "soccer",
    },
    "Bundesliga": {
        "url": "https://site.api.espn.com/apis/v2/sports/soccer/ger.1/standings?season={year}",
        "points_system": "soccer",
    },
    "Serie A": {
        "url": "https://site.api.espn.com/apis/v2/sports/soccer/ita.1/standings?season={year}",
        "points_system": "soccer",
    },
    "Ligue 1": {
        "url": "https://site.api.espn.com/apis/v2/sports/soccer/fra.1/standings?season={year}",
        "points_system": "soccer",
    },
}

YEARS = [2022, 2023, 2024, 2025]

OUTPUT_PATH = "/Users/corneliuswiehl/Documents/projects/startup/projects/luck_vs_skill/visualization/espn_data.json"


def get_stat(stats, name):
    """Extract a stat value by name from the ESPN stats list."""
    for s in stats:
        if s.get("name") == name:
            return s.get("value", 0)
    return 0


def calculate_points(wins, losses, ties, ot_losses, points_system):
    """Calculate points based on the league's points system."""
    if points_system == "nba":
        return wins * 2
    elif points_system == "nfl":
        return wins * 1 + ties * 0.5
    elif points_system == "nhl":
        return wins * 2 + ot_losses * 1
    elif points_system == "mlb":
        return wins * 1
    elif points_system == "soccer":
        return wins * 3 + ties * 1
    return 0


def fetch_standings(url):
    """Fetch standings JSON from ESPN API."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read().decode())


def parse_league_data(data, points_system):
    """Parse ESPN standings JSON into our desired format."""
    teams = []
    wins_list = []
    losses_list = []
    ties_list = []
    points_list = []
    games_played_set = set()

    # ESPN data has 'children' which are conferences/divisions
    # Each child has 'standings' -> 'entries'
    children = data.get("children", [])

    for child in children:
        standings = child.get("standings", {})
        entries = standings.get("entries", [])

        for entry in entries:
            team_name = entry["team"]["displayName"]
            stats = entry.get("stats", [])

            w = int(get_stat(stats, "wins"))
            l = int(get_stat(stats, "losses"))
            t = int(get_stat(stats, "ties"))

            # NHL has otLosses (overtime losses) which are separate from regular losses
            ot_losses = 0
            if points_system == "nhl":
                ot_losses = int(get_stat(stats, "otLosses"))

            # Get games played; compute if not available
            gp = get_stat(stats, "gamesPlayed")
            if gp:
                gp = int(gp)
            else:
                # For NBA/NFL/MLB, games = wins + losses + ties
                if points_system == "nhl":
                    gp = w + l + ot_losses
                else:
                    gp = w + l + t

            games_played_set.add(gp)

            pts = calculate_points(w, l, t, ot_losses, points_system)

            teams.append(team_name)
            wins_list.append(w)
            losses_list.append(l)
            ties_list.append(t)
            points_list.append(pts)

    # Determine a single "games" value (most common or max)
    if len(games_played_set) == 1:
        games = games_played_set.pop()
    else:
        # Use the most common value, or max as fallback
        games = max(games_played_set) if games_played_set else 0

    # Sort all lists by points descending
    combined = sorted(
        zip(teams, wins_list, losses_list, ties_list, points_list),
        key=lambda x: x[4],
        reverse=True,
    )

    if combined:
        teams, wins_list, losses_list, ties_list, points_list = zip(*combined)
        teams = list(teams)
        wins_list = list(wins_list)
        losses_list = list(losses_list)
        ties_list = list(ties_list)
        points_list = list(points_list)

    return {
        "teams": teams,
        "wins": wins_list,
        "losses": losses_list,
        "ties": ties_list,
        "games": games,
        "points": points_list,
    }


def main():
    all_data = {}
    total_requests = len(LEAGUES) * len(YEARS)
    completed = 0

    for league_name, league_config in LEAGUES.items():
        all_data[league_name] = {}

        for year in YEARS:
            url = league_config["url"].format(year=year)
            completed += 1
            print(f"[{completed}/{total_requests}] Fetching {league_name} {year}...", end=" ", flush=True)

            try:
                data = fetch_standings(url)
                parsed = parse_league_data(data, league_config["points_system"])
                all_data[league_name][str(year)] = parsed
                n_teams = len(parsed["teams"])
                print(f"OK ({n_teams} teams, {parsed['games']} games)")
            except Exception as e:
                print(f"ERROR: {e}")
                all_data[league_name][str(year)] = {
                    "teams": [],
                    "wins": [],
                    "losses": [],
                    "ties": [],
                    "games": 0,
                    "points": [],
                    "error": str(e),
                }

            # Be respectful - 1 second delay between requests
            time.sleep(1)

    # Save to JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nData saved to {OUTPUT_PATH}")
    print(f"File size: {len(json.dumps(all_data)):,} bytes")

    # Print summary
    print("\n=== SUMMARY ===")
    for league_name in all_data:
        print(f"\n{league_name}:")
        for year in sorted(all_data[league_name].keys()):
            season = all_data[league_name][year]
            n = len(season["teams"])
            if n > 0:
                top = season["teams"][0]
                top_pts = season["points"][0]
                print(f"  {year}: {n} teams, {season['games']} games | Top: {top} ({top_pts} pts)")
            else:
                err = season.get("error", "unknown error")
                print(f"  {year}: FAILED - {err}")


if __name__ == "__main__":
    main()
