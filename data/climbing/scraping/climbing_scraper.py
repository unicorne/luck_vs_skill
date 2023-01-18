"""
This python file scrapes the IFSC website for all climbing results from 2007 to 2022.
https://www.ifsc-climbing.org/index.php/world-competition/last-result
"""


import time

import pandas as pd
import requests
import tqdm

EVENT_URL = (
    "https://components.ifsc-climbing.org/results-api.php?api=event_results&event_id={}"
)
RESULT_URL = "https://components.ifsc-climbing.org/results-api.php?api=event_full_results&result_url={}"

data = {
    "id": [],
    "name": [],
    "league_id": [],
    "league_season_id": [],
    "season_id": [],
    "starts_at": [],
    "ends_at": [],
    "location": [],
    "dcat_name": [],
    "discipline_kind": [],
    "category_name": [],
    "rank": [],
    "athlete_id": [],
    "firstname": [],
    "lastname": [],
    "paraclimbing_sport_class": [],
    "country": [],
    "category_round_id": [],
    "round_name": [],
    "score": [],
}

steps = 4000
for i in tqdm.tqdm(range(50, steps), total=steps):
    try:
        json_event = requests.get(EVENT_URL.format(i)).json()
        id = json_event["id"]
        name = json_event["name"]
        # filter out non-IFSC events
        if "IFSC" not in name:
            continue
        league_id = json_event["league_id"]
        league_season_id = json_event["league_season_id"]
        season_id = json_event["season_id"]
        starts_at = json_event["starts_at"]
        ends_at = json_event["ends_at"]
        location = json_event["location"]
        for event in json_event["d_cats"]:
            dcat_name = event["dcat_name"]
            discipline_kind = event["discipline_kind"]
            category_name = event["category_name"]
            event_results = requests.get(
                RESULT_URL.format(event["full_results_url"])
            ).json()
            for athlete in event_results["ranking"]:
                rank = athlete["rank"]
                athlete_id = athlete["athlete_id"]
                firstname = athlete["firstname"]
                lastname = athlete["lastname"]
                paraclimbing_sport_class = athlete["paraclimbing_sport_class"]
                country = athlete["country"]
                for round in athlete["rounds"]:
                    category_round_id = round["category_round_id"]
                    round_name = round["round_name"]
                    score = round["score"]
                    # append all variables to data
                    data["id"].append(id)
                    data["name"].append(name)
                    data["league_id"].append(league_id)
                    data["league_season_id"].append(league_season_id)
                    data["season_id"].append(season_id)
                    data["starts_at"].append(starts_at)
                    data["ends_at"].append(ends_at)
                    data["location"].append(location)
                    data["dcat_name"].append(dcat_name)
                    data["discipline_kind"].append(discipline_kind)
                    data["category_name"].append(category_name)
                    data["rank"].append(rank)
                    data["athlete_id"].append(athlete_id)
                    data["firstname"].append(firstname)
                    data["lastname"].append(lastname)
                    data["paraclimbing_sport_class"].append(paraclimbing_sport_class)
                    data["country"].append(country)
                    data["category_round_id"].append(category_round_id)
                    data["round_name"].append(round_name)
                    data["score"].append(score)
        time.sleep(0.05)
    except requests.exceptions.JSONDecodeError as e:
        continue
    except Exception as e:
        print(e)
        continue


df = pd.DataFrame(data)
df.to_csv("../climbing_data.csv", index=False)
