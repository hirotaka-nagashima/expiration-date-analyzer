import abc
import csv
import datetime as dt
import glob
import os
from collections.abc import Iterable
from typing import Callable

import pandas as pd
from pandas.api import types

# Used mainly for reducing memory usage. Therefore, we can not type as
# Callable[[pd.DataFrame], pd.DataFrame].
Filter = Callable[[pd.DataFrame], None]


class FileIO(abc.ABC):
    """Interface handling files."""

    @abc.abstractmethod
    def read_tweets(self, index_col=None, filter_: Filter | None = None) -> pd.DataFrame | None:
        pass

    @abc.abstractmethod
    def read_dynamics(self, index_col=None, filter_: Filter | None = None) -> pd.DataFrame | None:
        pass

    @abc.abstractmethod
    def read_retweets(self, index_col=None, filter_: Filter | None = None) -> pd.DataFrame | None:
        pass

    @abc.abstractmethod
    def read_relationship(
        self, index_col=None, filter_: Filter | None = None
    ) -> pd.DataFrame | None:
        pass

    @abc.abstractmethod
    def log_tweets(self, tweets):
        pass

    @abc.abstractmethod
    def log_dynamics(self, tweets):
        pass

    @abc.abstractmethod
    def log_retweets(self, tweets):
        pass

    @abc.abstractmethod
    def log_relationship(self, shown_at, ids, to=None, from_=None):
        pass

    @abc.abstractmethod
    def write(self, name, df):
        pass


class CSVHandler(FileIO):
    """Handles csv files."""

    def __init__(self, data_dir):
        self.DATA_DIR = os.path.normpath(data_dir)
        self.ENCODING = "utf-8-sig"  # for Japanese

        # An element in the lists is used as a column name, key of other tables
        # such as self.DTYPE. For some items, it also corresponds to a name of
        # an instance variable of the Tweet object.
        self.HEADER = {
            "tweets": [
                "id",
                "shown_at",  # primary key
                "created_at",
                "full_text",
                "has_hashtags",
                "has_media",
                "has_urls",
                "has_user_mentions",
                "has_symbols",
                "has_polls",
                "user_id",
                "user_screen_name",
                "user_followers_count",
                "user_friends_count",
                "user_verified",
                "user_statuses_count",
                "retweet_count",
                "favorite_count",
            ],  # for Tweet objects
            "dynamics": [
                "id",
                "elapsed_time",  # primary key
                "user_followers_count",
                "user_friends_count",
                "user_statuses_count",
                "retweet_count",
                "favorite_count",
            ],  # for Tweet objects
            "retweets": [
                "id",  # primary key
                "retweeted_id",
                "created_at",
                "user_id",
                "user_screen_name",
                "user_followers_count",
                "user_friends_count",
                "user_verified",
                "user_statuses_count",
            ],  # for Tweet objects
            "relationship": ["shown_at", "to", "from"],
        }

        dtype = {
            # for Tweet objects
            "created_at": "str",
            "id": "category",
            "full_text": "str",
            "has_hashtags": "bool",
            "has_media": "bool",
            "has_urls": "bool",
            "has_user_mentions": "bool",
            "has_symbols": "bool",
            "has_polls": "bool",
            "user_id": "category",
            "user_screen_name": "str",
            "user_followers_count": "uint32",  # <= 350M (active users)
            "user_friends_count": "uint32",  # <= 350M
            "user_verified": "bool",
            "user_statuses_count": "uint32",  # <= 38M (@VENETHIS)
            "retweet_count": "uint32",  # <= 350M
            "favorite_count": "uint32",  # <= 350M
            "retweeted_id": "category",
            "shown_at": "str",
            "elapsed_time": "uint32",
            # others
            "to": "category",
            "from": "category",
        }
        parse_dates = ["created_at", "shown_at"]

        # Categorize them based on the self.HEADER.
        names = ["tweets", "dynamics", "retweets", "relationship"]

        def filter_dict(dict_, keys):
            return {k: dict_[k] for k in keys}

        self.DTYPE = {n: filter_dict(dtype, self.HEADER[n]) for n in names}
        self.PARSE_DATES = {n: list(set(parse_dates) & set(self.HEADER[n])) for n in names}

    def read_tweets(self, index_col=None, filter_: Filter | None = None) -> pd.DataFrame | None:
        return self._read("tweets", index_col, filter_)

    def read_dynamics(self, index_col=None, filter_: Filter | None = None) -> pd.DataFrame | None:
        return self._read("dynamics", index_col, filter_)

    def read_retweets(self, index_col=None, filter_: Filter | None = None) -> pd.DataFrame | None:
        return self._read("retweets", index_col, filter_)

    def read_relationship(
        self, index_col=None, filter_: Filter | None = None
    ) -> pd.DataFrame | None:
        return self._read("relationship", index_col, filter_)

    def _read(self, name, index_col=None, filter_: Filter | None = None) -> pd.DataFrame | None:
        # Get all paths for split files.
        filename = f"{name}_*.csv"
        srcs = glob.glob(os.path.join(self.DATA_DIR, filename))
        if not srcs:
            return

        # Read data from each file.
        dfs = []
        for src in srcs:
            df = pd.read_csv(
                src,
                dtype=self.DTYPE[name],
                parse_dates=self.PARSE_DATES[name],
                encoding=self.ENCODING,
            )
            if filter_ is not None:
                filter_(df)
            dfs.append(df)

        # Concatenate them.
        if len(dfs) == 1:
            concatenated_df = dfs[0]
        else:
            # Combine categoricals.
            for cname, dtype in self.DTYPE[name].items():
                if dtype == "category":
                    union = types.union_categoricals([df[cname] for df in dfs])
                    for df in dfs:
                        df[cname] = pd.Categorical(df[cname], categories=union.categories)
            concatenated_df = pd.concat(dfs)

        if index_col is not None:
            concatenated_df.set_index(index_col, drop=False, inplace=True)
            concatenated_df.sort_index(inplace=True)
        return concatenated_df

    def log_tweets(self, tweets):
        name = "tweets"
        rows = CSVHandler._get_rows(tweets, self.HEADER[name])
        self._log(name, rows)

    def log_dynamics(self, tweets):
        name = "dynamics"
        rows = CSVHandler._get_rows(tweets, self.HEADER[name])
        self._log(name, rows)

    def log_retweets(self, tweets):
        name = "retweets"
        rows = CSVHandler._get_rows(tweets, self.HEADER[name])
        self._log(name, rows)

    def log_relationship(self, shown_at, ids, to=None, from_=None):
        name = "relationship"
        rows = (
            [[shown_at, to, i] for i in ids]
            if from_ is None
            else [[shown_at, i, from_] for i in ids]
        )
        self._log(name, rows)

    def _log(self, name, rows):
        # Split logs into multiple files based on date.
        date = dt.datetime.today().strftime("%Y%m%d")
        filename = f"{name}_{date}.csv"
        dest = os.path.join(self.DATA_DIR, filename)

        os.makedirs(self.DATA_DIR, exist_ok=True)
        is_new_file = not os.path.exists(dest)
        with open(dest, "a", newline="", encoding=self.ENCODING) as file:
            writer = csv.writer(file)
            if is_new_file:
                writer.writerow(self.HEADER[name])
            writer.writerows(rows)

    @staticmethod
    def _get_row(tweet, mask: Iterable[str]):
        return [getattr(tweet, name) for name in mask]

    @staticmethod
    def _get_rows(tweets, mask: Iterable[str]):
        return [CSVHandler._get_row(t, mask) for t in tweets]

    def write(self, name, df):
        # Split logs into multiple files based on date.
        date = dt.datetime.today().strftime("%Y%m%d")
        filename = f"{name}_{date}.csv"
        dest = os.path.join(self.DATA_DIR, filename)

        os.makedirs(self.DATA_DIR, exist_ok=True)
        df.to_csv(dest, encoding=self.ENCODING, index=False)
