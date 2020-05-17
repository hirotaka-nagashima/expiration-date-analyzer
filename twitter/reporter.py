import datetime as dt
import inspect
import os
import sys
from typing import Optional

import tweepy

from twitter import api
from twitter import error
from twitter import tweet


class Reporter:
    """Class to report running status and errors by tweeting."""

    def __init__(self, api_):
        self._api = api_  # type: api.API
        self._parent_id = None  # type: Optional[tweet.ID]

    def report_beginning(self, status="Now running."):
        """Tweets a beginning and saves its ID for threading."""
        self._tweet(status)

    def report_finish(self):
        self._tweet("Done.")

    def report(self, status):
        self._tweet(status)

    def _tweet(self, content):
        """Tweets the content with the time and the PID.

        Tweets the content with the time and the PID to avoid the 187 error:
        "Status is a duplicate".
        """
        now = dt.datetime.now().replace(microsecond=0)
        pid = os.getpid()
        header = f"{now} (PID: {pid})"
        status = f"{header}\n{content}"
        try:
            if self._parent_id is None:
                # Tweet then save its tweet ID.
                self._parent_id = self._api.update_status(status=status).id
            else:
                # Reply to the tweet having the self._parent_id.
                self._api.update_status(status=status,
                                        in_reply_to_status_id=self._parent_id)
        except (error.TotalRateLimitError, tweepy.TweepError) as e:
            # NOTE: Do not call self._tweet_error() which can cause an infinite
            # loop.
            print(status)
            print(e)

    def report_error(self, e, **vars_):
        """Tweets the error and information to debug.

        Args:
            e: Error.
            vars_: Variables dumped in the tweet.
        """
        self._tweet_error(e, **vars_)

    def _tweet_error(self, e, **vars_):
        # NOTE: We should consider to include an icon in the comment for
        # visibility.
        if type(e) is error.TotalRateLimitError:
            comment = "Total limit has been reached."
        elif type(e) is tweepy.TweepError:
            comment = "TweepError occurred."
        else:
            # NOTE: Do not delete the space.
            comment = "⚠️Unexpected error occurred."

        # Add information to vars_.
        called_by = inspect.stack()[2].function
        vars_["function"] = called_by
        _, _, tb = sys.exc_info()
        lineno = tb.tb_lineno
        vars_["lineno"] = lineno

        detail = f"Detail: {e}"
        variables = f"Variables: {vars_}"
        content = f"{comment}\n{detail}\n{variables}"
        self._tweet(content)
