"""Extracts and normalizes time in a English sentence."""

import datetime as dt

import sutime
from dateutil import parser, relativedelta

from utils.timenormalizer import Duration, TimeExpressions

_tagger: sutime.SUTime | None = None


def load():
    """Creates an instance of a temporal tagger."""
    global _tagger
    _tagger = sutime.SUTime()


def _extract_time(sentence, doc_time: dt.datetime | None = None) -> TimeExpressions:
    """Extracts time expressions from a sentence.

    Args:
        sentence: Sentence to be extracted time expressions.
        doc_time: Time that the sentence created. Given None, the current time
            is used by SUTime.

    Returns:
        Time that each time-dependent keyword expresses.
    """
    reference_date = "" if doc_time is None else doc_time.isoformat()
    json_ = _tagger.parse(sentence, reference_date)

    def to_duration(timex_value: str) -> Duration:
        """Converts TIMEX3 to a duration.

        For example, given "2019-12-30", returns
            (dt.datetime(2019, 12, 30, 0, 0),
             dt.datetime(2019, 12, 30, 23, 59, 59, 999999)).
        However, given with the time like "2019-12-30T00", since it has no
        duration, returns
            (dt.datetime(2019, 12, 30, 0, 0), None).
        """

        def increment(raw_date: str) -> dt.datetime:
            delta = [
                relativedelta.relativedelta(years=1),
                relativedelta.relativedelta(months=1),
                relativedelta.relativedelta(days=1),
            ]
            index = len(raw_date.split("-")) - 1
            return parser.isoparse(raw_date) + delta[index]

        min_ = relativedelta.relativedelta(microseconds=1)

        since = parser.isoparse(timex_value)
        until = None if "T" in timex_value else increment(timex_value) - min_
        return since, until

    # Format the result.
    result = []
    datetime_list = json_
    for datetime_info in datetime_list:
        keyword = datetime_info["text"]
        try:
            time = to_duration(datetime_info["timex-value"])
        except ValueError:
            continue
        time_expression = (keyword, time)
        result.append(time_expression)
    return result


def extract_time(sentence, doc_time: dt.datetime | None = None) -> TimeExpressions:
    """Extracts time expressions from a sentence.

    Args:
        sentence: Sentence to be extracted time expressions.
        doc_time: Time that the sentence created. Given None, the current time
            is used by SUTime.

    Returns:
        Time that each time-dependent keyword expresses.
    """
    return _extract_time(sentence, doc_time)
