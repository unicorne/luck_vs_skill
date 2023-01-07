"""
This python file scrapes the GosuGamers website for League of Legends, Dota2 and CounterStrike Global Offensive results.    
"""
import requests
import pandas as pd
import tqdm
import time
from bs4 import BeautifulSoup

CSGO_URL = "https://www.gosugamers.net/counterstrike/matches/list/results?maxResults=100&page={}"
LOL_URL = "https://www.gosugamers.net/lol/matches/list/results?maxResults=100&page={}"
DOTA_URL = (
    "https://www.gosugamers.net/dota2/matches/list/results?maxResults=100&page={}"
)
steps = 4500


def scrape(URL, filename):
    # iterate through pages
    for i in tqdm.tqdm(range(0, steps), total=steps):
        data = {"match_date": [], "team": [], "score": [], "tournament": [], "win": []}
        response = requests.get(URL.format(i))
        # parse html
        bs = BeautifulSoup(response.content, "html.parser")
        match_list = bs.find("div", {"class": "match-list"})
        # iterate through matches
        for match in match_list.findChildren(recursive=False)[:-1]:
            # get match date
            if match.has_attr("class") and "match-date" in match["class"]:
                match_date = match.text.strip()
                continue
            # get match info
            elif (
                match.has_attr("class")
                and "cell" in match["class"]
                and len(match["class"]) == 1
            ):
                info = match.find("div", {"class": "match-info"})
                team1 = info.find("span", {"class": "team-1"}).text.strip()
                team2 = info.find("span", {"class": "team-2"}).text.strip()
                tournament = info.find(
                    "div", {"class": "match-tournament"}
                ).text.strip()
                score = match.find("div", {"class": "match-score"})
                spans = score.find_all("span")
                scores = [span.text.strip() for span in spans]
                classes = [span["class"][0] for span in spans]
                team1_score = scores[0]
                team2_score = scores[2]
                team1_win = classes[0]
                team2_win = classes[2]
                data["match_date"].append(match_date)
                data["team"].append(team1)
                data["score"].append(team1_score)
                data["tournament"].append(tournament)
                data["win"].append(team1_win)
                data["match_date"].append(match_date)
                data["team"].append(team2)
                data["score"].append(team2_score)
                data["tournament"].append(tournament)
                data["win"].append(team2_win)

        df = pd.DataFrame(data)
        df.to_csv(filename, mode="a", header=False, index=False)


scrape(CSGO_URL, "../data/raw_data/csgo_matches.csv")
scrape(LOL_URL, "../data/raw_data/lol_matches.csv")
scrape(DOTA_URL, "../data/raw_data/dota_matches.csv")
