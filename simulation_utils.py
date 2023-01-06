import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler


def simulate_league(
    df_schedule,
    probabilities_win_loss_tie,
    points_for_win_loss_tie,
    return_table=False,
    min_max_scaling=False,
):
    """
    Simulate a league with the given schedule and probabilities for win, loss and tie.


    Parameters
    ----------
    df_schedule : pandas.DataFrame
        The schedule of the league. Expects a DataFrame with columns "Home" and "Away" that represent the Home and Away team for each game.
    probabilities_win_loss_tie : list
        The probabilities for win, loss and tie.
    points_for_win_loss_tie : list
        The points for win, loss and tie.
    return_table : bool, optional
        If True, the function returns a table with the number of wins, losses and ties for each team. The default is False.
    min_max_scaling : bool, optional
        If True, the points are scaled to a range between 0 and 1. The default is False.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        If return_table is True, the function returns a table with the number of wins, losses and ties for each team.
        If return_table is False, the function returns an array with the points for each team.

    """

    df_schedule["Winner"] = np.random.choice(
        ["Home", "Away", "Tie"], size=len(df_schedule), p=probabilities_win_loss_tie
    )
    df_schedule["Outcome"] = df_schedule.apply(
        lambda x: x["Home"]
        if x["Winner"] == "Home"
        else (x["Away"] if x["Winner"] == "Away" else "Tie"),
        axis=1,
    )

    # Count the number of wins per team
    win_counter = Counter(df_schedule["Outcome"])

    # As ties are unlikely to happen, it is possible that no ties occurred in the simulation
    if "Tie" in win_counter:
        win_counter.pop("Tie")

    # count the number of ties per team
    tie_df = df_schedule[df_schedule["Outcome"] == "Tie"]
    tie_list = list(tie_df["Home"].values) + list(tie_df["Away"].values)
    tie_counter = Counter(tie_list)

    # count the number of losses per team by subtracting the number of wins and ties from the number of games played for each team
    loss_counter = (
        Counter(df_schedule["Home"].values)
        + Counter(df_schedule["Away"].values)
        - win_counter
        - tie_counter
    )

    simulated_results = pd.DataFrame(
        [win_counter, loss_counter, tie_counter], index=["Wins", "Loses", "Ties"]
    ).T
    # Replace NaN with 0 to avoid errors
    simulated_results.replace(np.nan, 0, inplace=True)
    simulated_results["Points"] = (
        points_for_win_loss_tie[0] * simulated_results["Wins"]
        + points_for_win_loss_tie[2] * simulated_results["Ties"]
        + points_for_win_loss_tie[1] * simulated_results["Loses"]
    )

    if return_table:
        return simulated_results
    else:
        if min_max_scaling:
            scaler = MinMaxScaler()
            simulated_results["Points"] = scaler.fit_transform(
                simulated_results["Points"].values.reshape(-1, 1)
            ).reshape(1, -1)[0]
        return simulated_results["Points"].values


def simulate_league_multiple_times(
    df_schedule,
    probabilities_win_loss_tie,
    points_for_win_loss_tie,
    n=100,
    return_table=False,
    min_max_scaling=False,
):
    """
    Simulate a league multiple times with the given schedule and probabilities for win, loss and tie.


    Parameters
    ----------
    df_schedule : pandas.DataFrame
        The schedule of the league. Expects a DataFrame with columns "Home" and "Away" that represent the Home and Away team for each game.
    probabilities_win_loss_tie : list
        The probabilities for win, loss and tie.
    points_for_win_loss_tie : list
        The points for win, loss and tie.
    n : int, optional
        The number of simulations. The default is 100.
    return_table : bool, optional
        If True, the function returns a table with the number of wins, losses and ties for each team. The default is False.
    min_max_scaling : bool, optional
        If True, the points are scaled to a range between 0 and 1. The default is False.

    Returns
    -------
    list of numpy.ndarray
        List of n simulated results of a league. Each simulation is a np.array of the point distribution for the teams.

    """
    simulated_results = [
        simulate_league(
            df_schedule,
            probabilities_win_loss_tie,
            points_for_win_loss_tie,
            return_table=return_table,
            min_max_scaling=min_max_scaling,
        )
        for i in range(n)
    ]
    return simulated_results


def calculate_variance_of_simulated_leagues(simulated_results):
    """
    Calculate the variance of the simulated results of a league.


    Parameters
    ----------
    simulated_results : list of numpy.ndarray
        List of n simulated results of a league. Each simulation is a np.array of the point distribution for the teams.

    Returns
    -------
    numpy.ndarray
        The variance of the simulated results of a league.

    """
    return np.mean([np.var(sim) for sim in simulated_results])
