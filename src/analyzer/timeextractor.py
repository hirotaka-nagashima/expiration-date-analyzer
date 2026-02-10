"""Extracts time expressions from tweets."""

import datetime as dt
import os
from typing import Dict, Optional

import requests
from dateutil import parser

from logger import fileio
from utils import jsonhandler, timenormalizer

_TweetID = str
AllTimeExpressions = Dict[_TweetID, timenormalizer.TimeExpressions]

FILENAME = "time.json"


def extract_time_expressions(data_dir, lang="ja") -> None:
    """Extracts time expressions from tweets and saves them."""
    path = _get_path(data_dir)
    data = jsonhandler.load(path)
    tweets_df = fileio.CSVHandler(data_dir).read_tweets()
    for id_, df in tweets_df.groupby("id", observed=True):
        if id_ in data:
            continue
        info = df.iloc[0].to_dict()
        sentence = info["full_text"]
        doc_time = info["created_at"] + dt.timedelta(hours=9)  # utc to jst
        try:
            tool = timenormalizer.ja if lang == "ja" else timenormalizer.en
            time_expressions = tool.extract_time(sentence, doc_time)
        except requests.HTTPError as e:
            print(f"{id_}: {e}")
            break
        data[id_] = time_expressions
    jsonhandler.dump(data, path)


def load_time_expressions(data_dir) -> AllTimeExpressions:
    path = _get_path(data_dir)
    data = jsonhandler.load(path)
    for raw_time_expressions in data.values():
        for index, raw_time_expression in enumerate(raw_time_expressions):

            def to_time_expression(
                keyword: str, since: str, until: Optional[str]
            ) -> timenormalizer.TimeExpression:
                """Casts a time expression loaded from a json."""
                since = parser.isoparse(since)
                until = None if until is None else parser.isoparse(until)
                return keyword, (since, until)

            raw_time_expressions[index] = to_time_expression(
                keyword=raw_time_expression[0],
                since=raw_time_expression[1][0],
                until=raw_time_expression[1][1],
            )
    return data


def _get_path(dir_):
    return os.path.join(os.path.normpath(dir_), FILENAME)
