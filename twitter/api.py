import collections
import datetime as dt
from typing import Deque
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import tweepy
import tzlocal

from twitter import error
from twitter import tweet


class API:
    """Wrapper of tweepy.API objects."""

    def __init__(self):
        # NOTE: Pay attention to revision of rate limits by Twitter.
        # https://developer.twitter.com/en/docs/basics/rate-limits
        self._api_selectors = {
            # POST statuses/update
            "update_status":
                TweepyAPISelector(user_auth_limit=25, app_auth_limit=0),
            
            # GET followers/ids
            "followers_ids":
                TweepyAPISelector(user_auth_limit=15, app_auth_limit=15),

            # GET search/tweets
            "search":
                TweepyAPISelector(user_auth_limit=180, app_auth_limit=450),

            # GET statuses/lookup
            "statuses_lookup":
                TweepyAPISelector(user_auth_limit=900, app_auth_limit=300),

            # GET statuses/retweets/:id
            "retweets":
                TweepyAPISelector(user_auth_limit=75, app_auth_limit=300),

            # GET statuses/show/:id
            "get_status":
                TweepyAPISelector(user_auth_limit=900, app_auth_limit=900),
        }

    @property
    def total_limit(self) -> Dict[str, int]:
        return {k: v.total_limit for k, v in self._api_selectors.items()}

    def append(self,
               api_user_auth: Optional[tweepy.API] = None,
               api_app_auth: Optional[tweepy.API] = None):
        for api_selector in self._api_selectors.values():
            api_selector.append(api_user_auth, api_app_auth)

    def update_status(self, **kwargs) -> tweet.Tweet:
        """POST statuses/update
        http://docs.tweepy.org/en/latest/api.html#API.update_status
        """
        return API._to_tweet(self._request("update_status", **kwargs))

    def followers_ids(self, **kwargs) -> List[tweet.ID]:
        """GET followers/ids
        http://docs.tweepy.org/en/latest/api.html#API.followers_ids
        """
        return self._request("followers_ids", **kwargs)

    def search(self, **kwargs) -> List[tweet.Tweet]:
        """GET search/tweets
        http://docs.tweepy.org/en/latest/api.html#API.search
        """
        return API._to_tweets(self._request("search", **kwargs))

    def statuses_lookup(self, **kwargs) -> List[tweet.Tweet]:
        """GET statuses/lookup
        http://docs.tweepy.org/en/latest/api.html#API.statuses_lookup
        """
        return API._to_tweets(self._request("statuses_lookup", **kwargs))

    def retweets(self, **kwargs) -> List[tweet.Tweet]:
        """GET statuses/retweets/:id
        http://docs.tweepy.org/en/latest/api.html#API.retweets
        """
        return API._to_tweets(self._request("retweets", **kwargs))

    def get_status(self, **kwargs) -> tweet.Tweet:
        """GET statuses/retweets/:id
        http://docs.tweepy.org/en/latest/api.html#API.get_status
        """
        return API._to_tweet(self._request("get_status", **kwargs))

    def _request(self, name, **kwargs):
        """
        Raises:
            error.TotalRateLimitError: Raised by the TweepyAPISelector before
                sending a request.
            tweepy.TweepError: Another error in the Twitter API. Not include
                the tweepy.RateLimitError because the error.TotalRateLimitError
                substitutes for it.
        """
        # Get an optimal tweepy.API.
        api_selector = self._api_selectors[name]
        api = api_selector.get_next_api()  # error.TotalRateLimitError?

        # Try to send a request.
        try:
            result = getattr(api, name)(**kwargs)  # tweepy.TweepError?
        except tweepy.RateLimitError:
            api_selector.log_rate_limit_error()
            return self._request(name, **kwargs)  # Try through a next API.
        else:
            api_selector.log_request()
            return result

    @staticmethod
    def _to_tweet(raw_tweet: tweepy.Status):
        return tweet.Tweet(raw_tweet, shown_at=dt.datetime.utcnow())

    @staticmethod
    def _to_tweets(raw_tweets: Iterable[tweepy.Status]):
        return [API._to_tweet(t) for t in raw_tweets]


class TweepyAPISelector:
    """Considers rate limiting and selects an optimal tweepy.API.

    This class considers rate limiting of the Twitter API and selects an optimal
    tweepy.API object. APIs are selected sequentially as they are concatenated.
    NOTE: One TweepyAPISelector must be assigned to one API function.
    """

    def __init__(self, user_auth_limit, app_auth_limit):
        self.USER_AUTH_LIMIT = user_auth_limit
        self.APP_AUTH_LIMIT = app_auth_limit
        self._total_limit = 0  # for lighter processing

        # [0] is always selected by rotating.
        self._apis = collections.deque()  # type: Deque[tweepy.API]
        self._limits = collections.deque()  # type: Deque[int]
        self._num_remains = collections.deque()  # type: Deque[int]
        self._reset_datetime = collections.deque()  # type: Deque[dt.datetime]

    @property
    def total_limit(self):
        return self._total_limit

    def _switch(self):
        # Next [0] is current [-1].
        self._apis.rotate(1)
        self._limits.rotate(1)
        self._num_remains.rotate(1)
        self._reset_datetime.rotate(1)

    @property
    def _current_api(self) -> tweepy.API:
        return self._apis[0]

    @property
    def _current_limit(self):
        return self._limits[0]

    @property
    def _next_limit(self):
        return self._limits[-1]

    @property
    def _current_num_remains(self):
        if self._current_reset_datetime <= dt.datetime.now():
            return self._current_limit
        return self._num_remains[0]

    @_current_num_remains.setter
    def _current_num_remains(self, num_remains):
        self._num_remains[0] = num_remains

    @property
    def _next_num_remains(self):
        if self._next_reset_datetime <= dt.datetime.now():
            return self._next_limit
        return self._num_remains[-1]

    @property
    def _current_reset_datetime(self):
        return self._reset_datetime[0]

    @_current_reset_datetime.setter
    def _current_reset_datetime(self, reset_datetime):
        self._reset_datetime[0] = reset_datetime

    @property
    def _next_reset_datetime(self):
        return self._reset_datetime[-1]

    @property
    def _current_limit_reached(self):
        return self._current_num_remains <= 0

    @property
    def _next_limit_reached(self):
        return self._next_num_remains <= 0

    @property
    def _current_break_secs(self):
        return 0 if not self._current_limit_reached else (
            self._current_reset_datetime - dt.datetime.now()).total_seconds()

    @property
    def _next_break_secs(self):
        return 0 if not self._next_limit_reached else (
            self._next_reset_datetime - dt.datetime.now()).total_seconds()

    def append(self,
               api_user_auth: Optional[tweepy.API] = None,
               api_app_auth: Optional[tweepy.API] = None):
        def append_api(api, limit):
            if api is not None:
                if 1 <= limit:
                    self._apis.append(api)
                    self._limits.append(limit)
                    self._num_remains.append(limit)
                    self._reset_datetime.append(dt.datetime.now())
                    self._total_limit += limit

        append_api(api_user_auth, self.USER_AUTH_LIMIT)
        append_api(api_app_auth, self.APP_AUTH_LIMIT)

    def get_next_api(self) -> tweepy.API:
        """
        Raises:
            error.TotalRateLimitError: Raised when all APIs get unavailable.
        """
        if self._total_limit <= 0:
            raise error.TotalRateLimitError(float("inf"))
        if self._current_limit_reached:
            if self._next_limit_reached:
                min_break_secs = min(self._current_break_secs,
                                     self._next_break_secs)
                raise error.TotalRateLimitError(min_break_secs)
            self._switch()
        return self._current_api

    def log_request(self):
        last_response = getattr(self._current_api, "last_response", None)
        if last_response is not None:
            headers = last_response.headers
            if "x-rate-limit-remaining" in headers:
                value = int(headers["x-rate-limit-remaining"])
                self._current_num_remains = value
            if "x-rate-limit-reset" in headers:
                value = int(headers["x-rate-limit-reset"])  # Unix timestamp
                self._current_reset_datetime = dt.datetime.fromtimestamp(
                    value, tzlocal.get_localzone()).replace(tzinfo=None)

    def log_rate_limit_error(self):
        self.log_request()
