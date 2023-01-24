import os

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

from src.simulation.simulation_utils import (
    calculate_variance_of_simulated_leagues,
    simulate_league,
    simulate_league_multiple_times,
)


def load_yaml():
    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def preprocess_observed_data(file_path, points_for_win_loss_tie):
    """
    Preprocess the observed data to calculate the variance of the points.

    Parameters
    ----------
    file_path : str
        The path to the observed data parquet file.
        The file should contain the columns "Year", "Wins", "Losses" and "Ties".
    points_for_win_loss_tie : list
        The points a team gets for a win, a loss and a tie.

    Returns
    -------
    pandas.DataFrame
        The preprocessed observed data.
    """

    # Load the data
    try:
        df = pd.read_parquet(file_path)
    except:
        raise ValueError("The file could not be loaded.")

    # Check if the dataframe has the right format
    if not (
        "Year" in df.columns,
        "Wins" in df.columns,
        "Losses" in df.columns,
        "Ties" in df.columns,
    ):
        raise ValueError(
            "The dataframe should contain the columns 'Year', 'Wins', 'Losses' and 'Ties'."
        )

    df["Points"] = (
        df["Wins"] * points_for_win_loss_tie[0]
        + df["Ties"] * points_for_win_loss_tie[2]
        + df["Losses"] * points_for_win_loss_tie[1]
    )
    df["Points_scaled"] = df["Points"].apply(
        lambda x: MinMaxScaler().fit_transform(np.array(x).reshape(-1, 1)).flatten()
    )
    df["Variance_observed_scaled"] = df["Points_scaled"].apply(lambda x: np.var(x))
    df["Variance_observed"] = df["Points"].apply(lambda x: np.var(x))
    df["#Teams"] = df["Teams"].apply(lambda x: len(x))

    return df


def calculate_observed_variance(df_preprocessed):
    """
    Calculate the variance of the points for the observed data.

    Parameters
    ----------
    df_preprocessed : pandas.DataFrame
        The preprocessed observed data.

    Returns
    -------
    float
        The variance of the points for the observed data.
    """

    if not ("Variance_observed" in df_preprocessed.columns):
        raise ValueError(
            "The dataframe should contain the columns 'Year', 'Wins', 'Losses' and 'Ties'."
        )

    return df_preprocessed["Variance_observed"].mean()


def get_available_schedules(directory):
    """
    Get the available schedules in the given directory.

    Parameters
    ----------
    directory : str
        The directory where the schedules are stored.

    Returns
    -------
    dict
        A dictionary with the available schedules.
    """

    available_schedules = {}
    nba_schedule_files = os.listdir(directory)

    for file in nba_schedule_files:
        if file.endswith(".csv"):
            available_schedules[file[:4]] = os.path.join(directory, file)

    return available_schedules


def calculate_luck_variance_per_year(
    df,
    available_schedules,
    probabilities_win_loss_tie,
    points_for_win_loss_tie,
    number_of_simulations,
    min_max_scaling=True,
    type="teams",
):

    """
    Calculate the luck variance per year.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the observed data.
    available_schedules : dict
        A dictionary with the available schedules.
    probabilities_win_loss_tie : list
        The probabilities a team wins, loses or ties.
    points_for_win_loss_tie : list
        The points a team gets for a win, a loss and a tie.
    number_of_simulations : int
        The number of simulations.
    type : str
        Type of simulation. Wether to simulate teams or compitions.
    Returns
    -------
    pandas.DataFrame
        The dataframe containing the observed data with a new luck variance column that contains the luck variance for every year where a schedule is available.


    """

    df["Luck_variance"] = np.nan
    df.set_index("Year", inplace=True)

    for year in df.index.values:
        if not str(year) in available_schedules:
            print("The schedule for the year {} is not available.".format(year))
        else:
            print("The schedule for the year {} is available.".format(year))
            tmp_schedule = pd.read_csv(available_schedules[str(year)])
            tmp_simulated = simulate_league_multiple_times(
                tmp_schedule,
                probabilities_win_loss_tie,
                points_for_win_loss_tie,
                n=number_of_simulations,
                return_table=False,
                min_max_scaling=min_max_scaling,
                type=type,
            )
            tmp_variance = calculate_variance_of_simulated_leagues(tmp_simulated)
            df.loc[year, "Luck_variance"] = tmp_variance

    return df
