"""fileio.Filter generators."""

import pandas as pd

from analyzer import timeextractor
from logger import fileio

_Filter = fileio.Filter
_AllTimeExpressions = timeextractor.AllTimeExpressions


def by_time_dependency(all_time_expressions: _AllTimeExpressions) -> _Filter:
    time_dependent_ids = [k for k, v in all_time_expressions.items() if v]

    def func(df: pd.DataFrame):
        cname = "retweeted_id" if "retweeted_id" in df.columns else "id"
        query = f"{cname} in {time_dependent_ids}"
        df.query(query, inplace=True)
    return func
