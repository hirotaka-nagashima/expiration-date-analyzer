import abc
import datetime as dt
import os
import re
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import eel
import numpy as np
import pandas as pd
from dateutil import parser
from scipy import signal

from analyzer import filters
from analyzer import timeextractor
from logger import fileio
from utils import functions
from utils import jsonhandler
from utils import timenormalizer

_AllTimeExpressions = timeextractor.AllTimeExpressions
_TimeExpressions = timenormalizer.TimeExpressions

Label = Optional[dt.datetime]
Labels = Dict[str, Label]
_Label = Optional[str]
_Labels = Dict[str, _Label]


class Labeler(abc.ABC):
    """Base class for labeling."""

    @staticmethod
    @abc.abstractmethod
    def filename():
        pass

    @classmethod
    def path(cls, data_dir):
        return os.path.join(os.path.normpath(data_dir), cls.filename())

    @classmethod
    def load_labels(cls, data_dir, additional_extension=None) -> Labels:
        path = cls.path(data_dir)
        if additional_extension is not None:
            path += f".{additional_extension}"
        data = jsonhandler.load(path)  # type: _Labels
        return {k: None if v is None else parser.isoparse(v)
                for k, v in data.items()}

    @staticmethod
    @abc.abstractmethod
    def run(data_dir) -> None:
        """Labels tweets in data_dir then writes out them."""
        pass


class AutoLabeler(Labeler):
    """Class for automatic labeling."""

    @staticmethod
    def filename():
        return "time_estimated.json"

    REFERRED_COLUMN = "retweet_count"
    TOLERANCE = 60 * 60  # secs

    WIDTHS_MA = (300, 1, 1)  # type: Tuple[int, int, int]
    A_SIGMOID = 100  # type: float
    B_SIGMOID = 0.9  # type: float

    @staticmethod
    def run(data_dir) -> None:
        """Labels tweets in data_dir then writes out them."""
        all_time_expressions = timeextractor.load_time_expressions(data_dir)
        ids_to_be_labeled = {k for k, v in all_time_expressions.items() if v}
        ids_labeled = AutoLabeler.load_labels(data_dir).keys()
        ids_to_be_labeled -= ids_labeled
        if not ids_to_be_labeled:  # All tweets are labeled already.
            return

        # Load tweets including time expressions, and their dynamics.
        filter_ = filters.by_time_dependency(all_time_expressions)
        io = fileio.CSVHandler(data_dir)
        tweets_df = io.read_tweets(index_col="id", filter_=filter_)
        dynamics_df = io.read_dynamics(index_col="elapsed_time",
                                       filter_=filter_)

        # Label them.
        labels = {}  # type: Labels
        for id_, df in dynamics_df.groupby("id", observed=True):
            if id_ not in ids_to_be_labeled:
                continue

            # Estimate expiration datetime.
            if len(df) < max(AutoLabeler.WIDTHS_MA):
                expiration_datetime = None
            else:
                time_expressions = all_time_expressions[id_]
                created_at = tweets_df.at[id_, "created_at"].to_pydatetime()
                x, y, ddy = AutoLabeler._calculate_ddy_for_estimation(df)
                expiration_datetime = AutoLabeler._estimate_expiration_datetime(
                    x, y, ddy, time_expressions, created_at)

            labels[id_] = expiration_datetime

        dest = AutoLabeler.path(data_dir)
        jsonhandler.update(labels, dest)

    @staticmethod
    def _calculate_ddy_for_estimation(
        dynamics_df: pd.DataFrame,
        uses_savgol_filter=False,
        smoothing_widths: Tuple[int, int, int] = WIDTHS_MA,
        smoothing_orders: Optional[Tuple[int, int, int]] = None,
        referred_column: str = REFERRED_COLUMN
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            dynamics_df: Dynamics of a tweet.
            uses_savgol_filter: On False, a centered moving average is used. On
                True, a Savitzky-Golay filter is used.
            smoothing_widths: Smoothing widths for y, dy, ddy. When using a
                Savitzky-Golay filter, values must be odd numbers.
            smoothing_orders: Polynomial orders for y, dy, ddy for a
                Savitzky-Golay filter.
            referred_column: Name of a column to be used as y.
        """
        def moving_average(x_, order_diff):
            excludes_edges = True if order_diff == 0 else False
            return functions.centered_moving_average(
                x_, smoothing_widths[order_diff], excludes_edges)

        def savgol_filter(x_, order_diff):
            return signal.savgol_filter(
                x_, smoothing_widths[order_diff], smoothing_orders[order_diff])

        smooth = savgol_filter if uses_savgol_filter else moving_average

        x = dynamics_df.index.values.copy()
        x = np.insert(x, 0, 0)
        y = dynamics_df[referred_column].values.copy()
        y = np.insert(y, 0, 0)
        y = smooth(y, 0)
        dx = np.gradient(x)
        dy = np.gradient(y) / dx
        dy = smooth(dy, 1)
        ddy = np.gradient(dy) / dx ** 2
        ddy = smooth(ddy, 2)

        upper_bound = y.argmax() + 1
        x = x[:upper_bound]
        y = y[:upper_bound]
        ddy = ddy[:upper_bound]
        return x, y, ddy

    @staticmethod
    def _estimate_expiration_datetime(
        x: np.ndarray,
        y: np.ndarray,
        ddy: np.ndarray,
        time_expressions: _TimeExpressions,
        created_at: dt.datetime,
        a_sigmoid: float = A_SIGMOID,
        b_sigmoid: float = B_SIGMOID,
        tolerance: float = TOLERANCE,
        refers_y=False
    ) -> Label:
        if refers_y:
            b_sigmoid = np.argwhere(y[-1] * b_sigmoid <= y)[0][0] / len(y)
        evaluation_value = -ddy * functions.sigmoid(
            a_sigmoid * (x / x[-1] - b_sigmoid))
        estimated_expiration_elapsed_time = x[evaluation_value.argmax()]

        # Approximate the estimated time as near time in time_expressions.
        for _, (since, until) in time_expressions:
            datetime = since if until is None else until
            delta = datetime - (created_at + dt.timedelta(hours=9))
            elapsed_time = delta.total_seconds()
            error = abs(estimated_expiration_elapsed_time - elapsed_time)
            if error <= tolerance:
                return datetime
        return None

    @staticmethod
    def try_with_various_parameters(
        data_dir,
        bounds: List[Tuple[float, float]],
        division_number: int
    ):
        """Calculates confusion matrices for each set of parameters."""
        all_time_expressions = timeextractor.load_time_expressions(data_dir)
        time_labeled = HandLabeler.load_labels(data_dir)

        # Load tweets including time expressions, and their dynamics.
        filter_ = filters.by_time_dependency(all_time_expressions)
        io = fileio.CSVHandler(data_dir)
        tweets_df = io.read_tweets(index_col="id", filter_=filter_)
        dynamics_df = io.read_dynamics(index_col="elapsed_time",
                                       filter_=filter_)

        # Calculate confusion matrices for each set of parameters.
        num_set_params = division_number ** len(bounds)
        confusion_matrices = [[0, 0, 0, 0] for _ in range(num_set_params)]
        tp_time = [0 for _ in range(num_set_params)]
        for id_, df in dynamics_df.groupby("id", observed=True):
            AutoLabeler._try_with_various_parameters(
                df,
                all_time_expressions[id_],
                time_labeled[id_],
                tweets_df.at[id_, "created_at"].to_pydatetime(),
                bounds, division_number,
                confusion_matrices, tp_time)

        # Show the result.
        print("TPt\tTP\tFP\tFN\tTN")
        for i in range(num_set_params):
            print(tp_time[i], *confusion_matrices[i], sep="\t")

    @staticmethod
    def _try_with_various_parameters(
        dynamics_df: pd.DataFrame,
        time_expressions: _TimeExpressions,
        time_labeled: Label,
        created_at: dt.datetime,
        bounds: List[Tuple[float, float]],
        division_number: int,
        confusion_matrices: List[List[int]],
        tp_time: List[int]
    ):
        if dynamics_df.empty:
            return

        prev_p2 = None
        x, y, ddy = None, None, None

        num_set_params = division_number ** len(bounds)
        for i in range(num_set_params):
            # Calculate current parameters from i.
            p = []
            for j, (lower, upper) in enumerate(bounds):
                # 0 or 1 or ... or (division_number - 1)
                i_ = int(i / division_number ** j) % division_number
                param = lower + (upper - lower) * (i_ / division_number)
                if j == 2:
                    param = int(param)
                p.append(param)

            if len(dynamics_df) < p[2]:
                prev_p2 = p[2]
                continue

            # Estimate expiration datetime.
            if prev_p2 is None or p[2] != prev_p2:  # for lighter processing
                x, y, ddy = AutoLabeler._calculate_ddy_for_estimation(
                    dynamics_df, smoothing_widths=(p[2], p[2], p[2]))
                prev_p2 = p[2]
            expiration_datetime = AutoLabeler._estimate_expiration_datetime(
                x, y, ddy, time_expressions, created_at,
                a_sigmoid=p[0], b_sigmoid=p[1])

            # Update a current confusion matrix.
            # tp: 0, fp: 1, fn: 2, tn: 3
            index_in_confusion_matrix = (
                (expiration_datetime is None) * 2 + (time_labeled is None))
            confusion_matrices[i][index_in_confusion_matrix] += 1
            if index_in_confusion_matrix == 0:  # tp
                if expiration_datetime == time_labeled:
                    tp_time[i] += 1


class HandLabeler(Labeler):
    """Helps a user with labeling of tweets on GUI.

    A user can choose explicit expiration datetime of a tweet as follows.
        1a. Mouse pointer + Left-clicking
        1b. Mouse wheel + Middle-clicking
    If there is not explicit expiration datetime, skip the tweet by
        2a. Left-clicking out of the tweet.
        2b. Middle-clicking before rotating the mouse wheel.
    """

    @staticmethod
    def filename():
        return "time_labeled.json"

    _HTML_DIR = r"C:\Users\Admin\Cloud\Programs\IntelliJ IDEA\Study\Best By Dates Analyzer\src\analyzer\labeler"
    _HTML_FILENAME = "index.html"

    _index = None  # type: Optional[int]
    _tweets_df = None  # type: Optional[pd.DataFrame]
    _all_time_expressions = None  # type: Optional[_AllTimeExpressions]
    _labels = None  # type: Optional[_Labels]

    @staticmethod
    def run(data_dir) -> None:
        """Opens a window for labeling of tweets in data_dir.

        Opens a window for labeling of tweets in data_dir. Note that we could
        not run simultaneously because of eel.
        """
        if HandLabeler._index is not None:  # Already running.
            return

        path = HandLabeler.path(data_dir)
        HandLabeler._index = -1
        HandLabeler._all_time_expressions = (
            timeextractor.load_time_expressions(data_dir))
        HandLabeler._labels = jsonhandler.load(path)

        # Load tweets including time expressions.
        filter_ = filters.by_time_dependency(HandLabeler._all_time_expressions)
        io = fileio.CSVHandler(data_dir)
        HandLabeler._tweets_df = io.read_tweets(filter_=filter_)

        # Open a window then pass the initiative.
        eel.init(HandLabeler._HTML_DIR)
        try:
            eel.start(HandLabeler._HTML_FILENAME)
        except (SystemExit, MemoryError, KeyboardInterrupt):
            jsonhandler.dump(HandLabeler._labels, path)
            HandLabeler._index = None
            HandLabeler._tweets_df = None
            HandLabeler._all_time_expressions = None
            HandLabeler._labels = None

    @staticmethod
    @eel.expose("askPythonLabel")
    def _label(expiration_datetime: _Label):
        current_tweet_id = HandLabeler._current_tweet()["id"]
        HandLabeler._labels[current_tweet_id] = expiration_datetime

    @staticmethod
    @eel.expose("askPythonProceed")
    def _proceed():
        # Get a next index.
        while True:
            HandLabeler._index += 1
            if len(HandLabeler._tweets_df) <= HandLabeler._index:
                raise SystemExit
            if HandLabeler._current_tweet()["id"] in HandLabeler._labels:
                continue  # Skip a labeled tweet.
            break

        next_tweet = HandLabeler._current_tweet()
        text = next_tweet["full_text"]
        text = re.sub(r"https://t\.co/\w+", "", text)

        # Supplement datetime to time expressions in the next_tweet.
        html = text
        id_ = next_tweet["id"]
        time_expressions = HandLabeler._all_time_expressions[id_]
        for keyword, (since, until) in time_expressions:
            datetime = (since if until is None else until).isoformat()
            start_tag = f"<time datetime=\"{datetime}\" tabindex=\"0\">"
            end_tag = "</time>"
            html = re.sub(f"((^|{end_tag})[^<]*?){re.escape(keyword)}",
                          f"\\1{start_tag}{keyword}{end_tag}", html)

        progress = HandLabeler._index / len(HandLabeler._tweets_df)
        created_at = str(next_tweet["created_at"] + dt.timedelta(hours=9))
        getattr(eel, "ask_js_proceed")(progress, created_at, html)

    @staticmethod
    def _current_tweet() -> pd.Series:
        return HandLabeler._tweets_df.iloc[HandLabeler._index]
