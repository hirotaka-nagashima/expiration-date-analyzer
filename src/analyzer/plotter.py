"""Graphs dynamics of tweets."""

import datetime as dt
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize

from analyzer import filters, timeextractor
from analyzer.labeler import labeler
from logger import fileio
from utils import functions, timenormalizer

_AllTimeExpressions = timeextractor.AllTimeExpressions
_TimeExpressions = timenormalizer.TimeExpressions

LOG_ENCODING = "utf-8-sig"  # for Japanese
LOG_FILENAME = "log.txt"

plt.style.use("ggplot")


def show_dynamics(data_dir, shows_retweet, shows_favorite, logscale=False, exports_to=None):
    all_time_expressions = timeextractor.load_time_expressions(data_dir)
    time_labeled = labeler.HandLabeler.load_labels(data_dir)

    # Load tweets including time expressions, and their dynamics.
    filter_ = filters.by_time_dependency(all_time_expressions)
    io = fileio.CSVHandler(data_dir)
    tweets_df = io.read_tweets(index_col="id", filter_=filter_)
    dynamics_df = io.read_dynamics(index_col="elapsed_time", filter_=filter_)

    for i, (id_, df) in enumerate(dynamics_df.groupby("id", observed=True)):
        show_dynamics_of(
            tweets_df.loc[id_],
            df,
            shows_retweet,
            shows_favorite,
            logscale=logscale,
            exports_to=exports_to,
            shortened_id=i,
            time_expressions=all_time_expressions[id_],
            label=time_labeled[id_],
        )


def show_dynamics_of(
    tweet: pd.Series,
    dynamics_df: pd.DataFrame,
    shows_retweet,
    shows_favorite,
    logscale=False,
    exports_to=None,
    shortened_id=None,
    time_expressions: _TimeExpressions | None = None,
    label: labeler.Label = None,
):
    # -1:, 0: RT, 1: FAV, 2: RT&FAV
    mode = shows_retweet + shows_favorite * 2 - 1
    if mode == -1 or dynamics_df.empty:
        return

    # Plot the data.
    df = dynamics_df.set_index("elapsed_time")  # also to copy
    df.index /= 60 * 60  # seconds to hours
    columns = [["retweet_count"], ["favorite_count"], ["favorite_count", "retweet_count"]][mode]
    df = df.loc[:, columns]
    df.loc[0] = 0
    df.sort_index(inplace=True)
    df.plot()

    # Highlight the time extracted from the tweet.
    created_at = tweet["created_at"].to_pydatetime()
    if time_expressions is not None:
        for _, (since, until) in time_expressions:
            _highlight_time(created_at, since, until)
    if label is not None:
        _highlight_time(created_at, label, color="r")

    id_ = tweet["id"] if shortened_id is None else shortened_id
    if logscale:
        ax = plt.gca()
        ax.set_yscale("log")
        ax.set_xscale("log")
    plt.title(f"Tweet {id_}")
    plt.xlabel("Time (hours)")
    plt.ylabel(["Retweet Count", "Favorite Count", ""][mode])
    plt.legend()

    # Show/Export the data.
    log = f"{id_}: {tweet['full_text']}\n"
    if exports_to is None:
        print(log, end="")
        plt.show()
    else:
        dir_ = os.path.normpath(exports_to)
        os.makedirs(dir_, exist_ok=True)
        log_dest = os.path.join(dir_, LOG_FILENAME)
        with open(log_dest, "a", encoding=LOG_ENCODING) as file:
            file.write(log)
        plot_filename = f"{id_}.png"
        plot_dest = os.path.join(dir_, plot_filename)
        plt.savefig(plot_dest)
    plt.close()


def _highlight_time(
    created_at: dt.datetime, since: dt.datetime, until: dt.datetime | None = None, color="k"
):
    def to_elapsed_time(datetime):
        delta = datetime - (created_at + dt.timedelta(hours=9))
        return delta.total_seconds() / (60 * 60)

    since = to_elapsed_time(since)
    if until is None:
        until = since + 0.2
        alpha = 0.3
    else:
        until = to_elapsed_time(until)
        alpha = 0.1
    plt.gca().axvspan(since, until, facecolor=color, alpha=alpha)


def _plot_trendline(x, y):
    f = functions.power_law
    popt = optimize.curve_fit(f, x, y, bounds=(0, np.inf))
    label = _to_label(popt, ndigits=3, names=["a", "c"])
    plt.plot(x, f(x, *popt[0]), label=label)


def _plot_diff_trendline(x, y, with_exponential_cutoff):
    f = functions.power_law
    popt = optimize.curve_fit(f, x, y, bounds=([-np.inf, 0], [0, np.inf]))
    if with_exponential_cutoff:
        # Try curve fitting again from the calculated parameters.
        f = functions.power_law_with_exponential_cutoff
        popt = optimize.curve_fit(
            f,
            x,
            y,
            p0=[popt[0][0], 0, popt[0][1]],
            bounds=([-np.inf, -np.inf, 0], [0, 0, np.inf]),
            maxfev=100000,
        )
        label = _to_label(popt, ndigits=3, names=["a", "b", "c"])
        plt.plot(x, f(x, *popt[0]), label=label)
    else:
        label = _to_label(popt, ndigits=3, names=["a", "c"])
        plt.plot(x, f(x, *popt[0]), label=label)


def _to_label(popt, ndigits, names):
    temp_labels = []
    for name, value in zip(names, popt[0]):
        shortened_value = round(value, ndigits)
        temp_label = f"{name}={shortened_value}"
        temp_labels.append(temp_label)
    label = ",".join(temp_labels)
    latex = f"${label}$"
    return latex
