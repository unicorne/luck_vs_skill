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
    type="teams",
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
    type : str
        Type of simulation. Wether to simulate teams or bouldering or climbing competitions. The default is "teams".

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
        if type == "teams"
        else simulate_competition(
            type,
            df_schedule=df_schedule,
            probabilities_win_loss_tie=probabilities_win_loss_tie,
            points_for_win_loss_tie=points_for_win_loss_tie,
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


def simulate_competition(type, **kwargs):
    """
    This functions decides which type of competition to simulate.

    Parameters
    ----------
    type : str
        Type of simulation. Wether to simulate teams or bouldering or climbing competitions. The default is "teams".

    Returns
    -------
    list of numpy.ndarray
        List of n simulated results of a league. Each simulation is a np.array of the point distribution for the teams.
    """

    if type == "bouldering":
        return simulate_boudlering_season(**kwargs)
    elif type == "climbing":
        return simulate_climbing_season(**kwargs)


def simulate_boudlering_season(
    df_schedule,
    probabilities_win_loss_tie,
    points_for_win_loss_tie,
    return_table=False,
    min_max_scaling=False,
):
    """
    Simulate a bouldering season with the given schedule and probabilities for win, loss and tie.

    Parameters
    ----------
    df_schedule : pandas.DataFrame
        The schedule of the league. Expects a DataFrame with columns "athlete_id" and "total_starter", "name" that represent the athlete and total starter for each competition.
    probabilities_win_loss_tie : list
        The probabilities for win, loss and tie. Not used in this function.
    points_for_win_loss_tie : list
        The points for win, loss and tie. Not used in this function.
    return_table : bool, optional
        If True, the function returns a table with the number of wins, losses and ties for each team. The default is False.
    min_max_scaling : bool, optional
        If True, the points are scaled to a range between 0 and 1. The default is False.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        If return_table is True, the function returns a table with the mean rank as Wins, losses and ties for each athlete.
        If return_table is False, the function returns an array with the points for each athlete.
    """

    athletes_points = {}
    for competition in df_schedule["name"].unique():
        ## for each athlete compute a random score
        athletes_in_competition = df_schedule[df_schedule["name"] == competition][
            "athlete_id"
        ].unique()
        # each athlete has to do 5 boulders. He can top out on 0-5 of them. Each boulder has also a zone, which is also ranked. And then for each top and zone the number of tries are counted.
        tops = np.random.randint(0, 6, size=len(athletes_in_competition))
        zones = np.random.randint(0, 6, size=len(athletes_in_competition))
        tops_tries = [np.random.randint(top, 10) for top in tops]
        # additional points for number of tries
        tops_tries_points = [
            (1 / top_tries) * 10 if top_tries != 0 else 0 for top_tries in tops_tries
        ]
        zones_tries = [np.random.randint(zone, 10) for zone in zones]
        # additional points for number of tries
        zones_tries_points = [
            (1 / zone_tries) * 10 if zone_tries != 0 else 0
            for zone_tries in zones_tries
        ]
        points = [
            (
                top,
                zone,
                top_tries,
                zone_tries,
                tops_tries_point,
                zones_tries_point,
                athlete,
            )
            for top, zone, top_tries, zone_tries, tops_tries_point, zones_tries_point, athlete in zip(
                tops,
                zones,
                tops_tries,
                zones_tries,
                tops_tries_points,
                zones_tries_points,
                athletes_in_competition,
            )
        ]
        # order by tops then by zones in descending, then by tries in ascending order
        points = sorted(points, key=lambda x: (-x[0], -x[1], x[2], x[3]))
        # compute the score
        scores = [
            (top * 10) + tops_tries_point + zone + zones_tries_point
            for top, zone, top_tries, zone_tries, tops_tries_point, zones_tries_point, athlete in points
        ]
        points = [p + (s,) for p, s in zip(points, scores)]

        # add the points to the athletes_points dictionary
        for (
            top,
            zone,
            top_tries,
            zone_tries,
            tops_tries_point,
            zones_tries_point,
            athlete,
            point,
        ) in points:
            if athlete not in athletes_points:
                athletes_points[athlete] = [point]
            else:
                athletes_points[athlete].append(point)

    # compute the mean score for each athlete
    data = []
    for athlete in athletes_points:
        data.append([athlete, np.mean(athletes_points[athlete])])

    df = pd.DataFrame(data, columns=["Teams", "Wins"])
    df["Losses"] = 0
    df["Ties"] = 0
    df["Points"] = df["Wins"]
    if return_table:
        return df
    else:
        if min_max_scaling:
            scaler = MinMaxScaler()
            df["Points"] = scaler.fit_transform(
                df["Points"].values.reshape(-1, 1)
            ).reshape(1, -1)[0]
        return df["Points"].values


def simulate_climbing_season(
    df_schedule,
    probabilities_win_loss_tie,
    points_for_win_loss_tie,
    return_table=False,
    min_max_scaling=False,
):
    """
    Simulate a league with the given schedule and probabilities for the rank of one athlete.
    For convinience we use the same names as in the other functions. Wins stands here for the rank of a athlete. Losses and Ties are not used.
    Parameters
    ----------
    df_schedule: pandas.DataFrame
        The schedule of the league. Expects a DataFrame with columns "athlete_id" and "total_starter", "name" that represent the athlete and total starter for each competition.
    probabilities_win_loss_tie : list
        The probabilities for win, loss and tie.
    points_for_win_loss_tie : list
        The points for win, loss and tie.
    return_table : bool, optional
        If True, the function returns a table with the mean ranks, losses and ties for each athlete. The default is False.
    min_max_scaling : bool, optional
        If True, the points are scaled to a range between 0 and 1. The default is False.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        If return_table is True, the function returns a table with the mean rank as Wins, losses and ties for each athlete.
        If return_table is False, the function returns an array with the points for each athlete.
    """

    # simulate one competition
    athletes_points = {}

    for competition in df_schedule["name"].unique():
        ## for each athlete compute a random rank
        athletes_in_competition = df_schedule[df_schedule["name"] == competition][
            "athlete_id"
        ].unique()
        total_starter = (
            df_schedule[df_schedule["name"] == competition]["total_starter"].unique()[0]
            + 1
        )  # +1 because we start at 1
        ranks = np.random.choice(
            np.arange(1, total_starter),
            size=len(athletes_in_competition),
            replace=False,
        )
        for athlete, rank in zip(athletes_in_competition, ranks):
            points = (total_starter - rank) / total_starter
            if athlete not in athletes_points:
                athletes_points[athlete] = [points]

            else:
                athletes_points[athlete].append(points)

    # compute mean rank for each athlete
    data = []
    for athlete in athletes_points:
        data.append([athlete, np.mean(athletes_points[athlete])])

    df = pd.DataFrame(data, columns=["Teams", "Wins"])
    df["Losses"] = 0
    df["Ties"] = 0
    df["Points"] = df["Wins"]
    if return_table:
        return df
    else:
        if min_max_scaling:
            scaler = MinMaxScaler()
            df["Points"] = scaler.fit_transform(
                df["Points"].values.reshape(-1, 1)
            ).reshape(1, -1)[0]
        return df["Points"].values
