import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import entropy

from src.simulation.simulation_utils import simulate_league_multiple_times
from src.utils.utils import load_yaml


def get_maximum_teams_games_combo(df):
    """
    Get the number of teams number of games combo with the most occurences in the data.


    Parameters
    ----------
    df : pandas.DataFrame
        The data. Expects a DataFrame with columns "#Teams" and "#Games".
    Returns
    -------
    int
        The maximum number of teams.
    int
        The maximum number of games.
    """

    index_val = (
        df.groupby(["#Teams", "#Games"])
        .count()
        .sort_values(by="Teams", ascending=False)
        .index[0]
    )
    teams = index_val[0]
    games = index_val[1]
    return teams, games


def get_probability_distribution_of_points(points):
    """
    Get the probability distribution of the points.


    Parameters
    ----------
    points : numpy.ndarray
        Numpy array of either observed or simulated points.


    Returns
    -------
    numpy.ndarray
        The probability distribution of the points.
    """

    config = load_yaml()
    bins = np.array(config["General"]["entropy_bins"])

    points_counter = Counter()
    points_counter.update({x: 0 for x in bins})
    points_counter.update(points)
    points_counter_values = np.array(list(points_counter.values()))
    points_counter_probabilities = points_counter_values / points_counter_values.sum()
    return points_counter_probabilities


def calculate_observed_entropy(df):
    """
    Calculate the entropy of the observed results of a sport/league.


    Parameters
    ----------
    df : pandas.DataFrame
        The observed data. Expects a DataFrame with columns "Points".

    Returns
    -------
    float
        The entropy of the distribution of the points.
    """
    points = np.concatenate(df["Points_binned"].values)
    points_counter_probabilities = get_probability_distribution_of_points(points)
    return entropy(points_counter_probabilities)


def calculate_simulated_distribution(df, sport):
    """
    Calculate the entropy of the simulated results of a sport/league.

    Simulates the season if a schedule is available and calculates the entropy based on the simulated point distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        The observed data. Expects a DataFrame with columns "Points" and Index "Year".

    Returns
    -------
    np.ndarray
        The probability distribution of the simulated points over the bin values
    """
    # Load config
    config = load_yaml()
    bins = np.array(config["General"]["entropy_bins"])

    # Simulate suitable seasons
    sims = []
    for year, ngames in zip(df.index, df["#Games"].values):
        try:
            df_schedule = pd.read_csv(
                os.path.join(config[sport]["schedule_directory"], f"{year}.csv")
            )
        except:
            print(f"No schedule available for year: {year}. Skipping this year.")
            continue
        # Simulate league
        tmp_sim = simulate_league_multiple_times(
            df_schedule,
            config[sport]["probabilities_win_loss_tie"],
            config[sport]["points_for_win_loss_tie"],
            config["General"]["number_of_simulations"],
        )
        tmp_sim = np.concatenate([x for x in tmp_sim])

        # Calculate
        maximal_possible_points = ngames * config[sport]["points_for_win_loss_tie"][0]
        tmp_sim = [x / maximal_possible_points for x in tmp_sim]
        tmp_sim = [bins[np.digitize(x, bins)] for x in tmp_sim]

        # Concatenate simulated points of seasons
        sims.append(tmp_sim)
    # Concatenate again across different years
    simulated_points = np.concatenate([x for x in sims])

    # Calculate simulated probability distribution
    tmp_counter_probabilities = get_probability_distribution_of_points(simulated_points)
    return tmp_counter_probabilities


def calculate_simulated_entropy(df, sport):
    """
    Calculate the entropy of the simulated results of a sport/league.

    Simulates the season if a schedule is available and calculates the entropy based on the simulated point distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        The observed data. Expects a DataFrame with columns "Points" and Index "Year".

    Returns
    -------
    float
        The entropy of the distribution of the simulated points.
    """
    tmp_counter_probabilities = calculate_simulated_distribution(df, sport)
    return entropy(tmp_counter_probabilities)


def calculate_entropy(path_to_prepared_data, sport):
    """
    Calculate the entropy of the observed and simulated results of a sport/league.

    Parameters
    ----------
    path_to_prepared_data : str
        Path to the prepared data.

    Returns
    -------
    pandas.DataFrame
        The dataframe used for calculation
    float
        The entropy of the distribution of the observed points.
    float
        The entropy of the distribution of the simulated points.

    """

    df = pd.read_parquet(path_to_prepared_data)
    df = calculate_binning(df, sport)

    teams, games = get_maximum_teams_games_combo(df)
    df_filtered = df[(df["#Teams"] == teams) & (df["#Games"] == games)]

    observed_entropy = calculate_observed_entropy(df_filtered)
    simulated_entropy = calculate_simulated_entropy(df_filtered, sport)

    return df, observed_entropy, simulated_entropy


def calculate_binning(df, sport):

    # Load the bins
    config = load_yaml()
    bins = np.array(config["General"]["entropy_bins"])

    # Calculate the % of maximal points possible
    maximal_points_possible = df["#Games"] * config[sport]["points_for_win_loss_tie"][0]
    df["Points_percentage"] = df["Points"] / maximal_points_possible

    # Bin the points
    # print(np.max([np.max(x) for x in df["Points_percentage"].values]))
    # print(df[:5])
    df["Points_binned"] = df["Points_percentage"].apply(
        lambda x: bins[np.digitize(x, bins)]
    )

    return df
