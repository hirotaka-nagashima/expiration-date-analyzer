"""Normalizes time in a sentence."""

import collections
import datetime as dt
import json
from typing import List
from typing import Optional
from typing import Tuple

import requests
from dateutil import parser
from dateutil import relativedelta

# NOTE: TimeExpressions should not be defined as Dict[str, Duration] because
# same words may express different time, which is specification of goo service.
Duration = Tuple[dt.datetime, Optional[dt.datetime]]
TimeExpression = Tuple[str, Duration]
TimeExpressions = List[TimeExpression]

_REQUEST_URL = "https://labs.goo.ne.jp/api/chrono"

# Defined as deque because used with rotating. At first, [0] is tried, then [-1]
# is tried as next [0] on an 400 error: "Rate limit exceeded". It makes next
# calls error-less.
_app_ids = collections.deque([])


def load_credentials(path):
    """Loads Goo labs API credentials."""
    with open(path) as file:
        raw_credentials = json.load(file)["credentials"]
    _app_ids.clear()
    for r in raw_credentials:  # each account
        _app_ids.append(r)


def _extract_time(sentence,
                  doc_time: Optional[dt.datetime] = None) -> TimeExpressions:
    """Extracts time expressions from a sentence.

    Args:
        sentence: Sentence to be extracted time expressions.
        doc_time: Time that the sentence created. Given None, the current time
            is used by goo.

    Returns:
        Time that each time-dependent keyword expresses.

    Raises:
        requests.HTTPError: Raised by goo service.
        IndexError: Raised when credentials are not loaded.
    """
    def request():
        """
        Raises:
            requests.HTTPError: Raised by goo service.
            IndexError: Raised when credentials are not loaded.
        """
        num_app_ids = len(_app_ids)

        # Prepare a json for a request.
        request_json = {"sentence": sentence}
        if doc_time is not None:
            request_json["doc_time"] = doc_time.isoformat()
        for num_unavailable_app_ids in range(num_app_ids):
            request_json["app_id"] = _app_ids[0]

            # Send a request.
            response = requests.post(_REQUEST_URL, json=request_json)
            response_json = json.loads(response.text)
            if response.status_code == 400:
                error_message = response_json["error"]["message"]
                if error_message == "Rate limit exceeded":
                    # Try with next app_id.
                    num_hopes = num_app_ids - (num_unavailable_app_ids + 1)
                    if 1 <= num_hopes:
                        _app_ids.rotate(1)
                        continue
            response.raise_for_status()
            return response_json
        raise IndexError("Load Goo labs API credentials.")

    json_ = request()

    def to_duration(goo_datetime: str) -> Duration:
        """Converts ISO 8601 datetime to a duration.

        For example, given "1901/2000", returns
            (dt.datetime(1901, 1, 1, 0, 0),
             dt.datetime(2000, 12, 31, 23, 59, 59, 999999)),
        given "2019-12-30", returns
            (dt.datetime(2019, 12, 30, 0, 0),
             dt.datetime(2019, 12, 30, 23, 59, 59, 999999)).
        However, given with the time like "2019-12-30T00", since it has no
        duration, returns
            (dt.datetime(2019, 12, 30, 0, 0), None).
        """
        def increment(raw_date: str) -> dt.datetime:
            delta = [relativedelta.relativedelta(years=1),
                     relativedelta.relativedelta(months=1),
                     relativedelta.relativedelta(days=1)]
            index = len(raw_date.split("-")) - 1
            return parser.isoparse(raw_date) + delta[index]
        min_ = relativedelta.relativedelta(microseconds=1)

        raw_datetime = goo_datetime.split("/")
        if len(raw_datetime) == 1:
            raw = raw_datetime[0]
            since = parser.isoparse(raw)
            until = None if "T" in raw else increment(raw) - min_
            return since, until
        else:
            since = parser.isoparse(raw_datetime[0])
            until = increment(raw_datetime[1]) - min_
            return since, until

    # Format the result.
    result = []
    datetime_list = json_["datetime_list"]
    for datetime_info in datetime_list:
        if "????" not in datetime_info[1]:
            keyword = datetime_info[0]
            time = to_duration(datetime_info[1])
            time_expression = (keyword, time)
            result.append(time_expression)
    return result


def extract_time(sentence,
                 doc_time: Optional[dt.datetime] = None) -> TimeExpressions:
    """Extracts time expressions from a sentence.

    Args:
        sentence: Sentence to be extracted time expressions.
        doc_time: Time that the sentence created. Given None, the current time
            is used by goo.

    Returns:
        Time that each time-dependent keyword expresses.

    Raises:
        requests.HTTPError: Raised by goo service.
    """
    return _extract_time(sentence, doc_time)


def is_time_dependent(sentence):
    """
    Raises:
        requests.HTTPError: Raised by goo service.
    """
    return bool(_extract_time(sentence))
