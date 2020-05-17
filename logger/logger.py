import collections
import datetime as dt
import itertools
import math
import time
from typing import Deque
from typing import Optional
from typing import Set

import requests
import tweepy

from logger import fileio
from twitter import api
from twitter import error
from twitter import reporter
from twitter import tweet


class Logger:
    """Logs tweets and tracks their dynamics."""

    def __init__(self, api_, io):
        self._api = api_  # type: api.API
        self._io = io  # type: fileio.FileIO

    def track_dynamics(self, loop_cycle, max_num_ids_to_track, additional_q="",
                       only_search=False, limits_min_retweets=True):
        """Tracks and logs various values, retweets for tweets searched for.

        Args:
            loop_cycle: Cycle[secs] to request information about tweets.
            max_num_ids_to_track: The maximum number of IDs to track
                simultaneously.
            additional_q: Additional query for "lang:ja min_retweets:50".
            only_search: Whether only searches for tweets.
            limits_min_retweets: Whether activates the query "min_retweets".
        """
        report_cycle = 60 * 60  # secs

        cycles = {}  # times (dimensionless)
        lcm_cycles = 1  # times (dimensionless), to reset a count
        def lcm(a, b): return (a * b) // math.gcd(a, b)
        for name, total_limit in self._api.total_limit.items():
            cycles[name] = math.ceil(15 * 60 / total_limit / loop_cycle)
            lcm_cycles = lcm(lcm_cycles, cycles[name])

        reporter_ = reporter.Reporter(self._api)
        reporter_.report_beginning(status="Now tracking dynamics.")

        count = 0
        prev_time = dt.datetime.now()
        prev_report_time = dt.datetime.now()
        logged_ids = set()
        logged_retweets_ids = set()
        ids_to_track_values = collections.deque()
        ids_to_track_retweets = collections.deque()
        ids_to_track_retweets_first = set()
        while True:
            # Search for tweets to track.
            if count % cycles["search"] == 0:
                # To collect retweets after, the min_retweets must be not more
                # than 100. Here, we set a tentative value but the best way is
                # probably that change the value by considering speed of the
                # timeline.
                q = "lang:ja "
                if limits_min_retweets:
                    q += "min_retweets:50 "
                q += additional_q
                num_ids_tracked_now = len(ids_to_track_values)
                max_count_to_log = max_num_ids_to_track - num_ids_tracked_now
                new_ids = self._search_tweets(q, logged_ids, reporter_,
                                              max_count_to_log=max_count_to_log)

                # Update IDs to track.
                ids_to_track_values.extend(new_ids)
                ids_to_track_retweets.extend(new_ids)
                ids_to_track_retweets_first |= new_ids

            # Track dynamics of tweets.
            if not only_search:
                if count % cycles["statuses_lookup"] == 0:
                    self._track_values(ids_to_track_values, reporter_)
                if count % cycles["retweets"] == 0:
                    id_given_up = self._track_retweets(
                        ids_to_track_retweets, ids_to_track_retweets_first,
                        logged_retweets_ids, reporter_)
                    if id_given_up is not None:
                        ids_to_track_values.remove(id_given_up)

            # Tweet running status.
            elapsed = (dt.datetime.now() - prev_report_time).total_seconds()
            if report_cycle <= elapsed:
                num_tweets = len(logged_ids)
                num_tracking_tweets = len(ids_to_track_retweets)
                percentage = int(100 * num_tracking_tweets / num_tweets)
                status = f"📢Tracking {percentage}% of {num_tweets}."
                reporter_.report(status)
                prev_report_time = dt.datetime.now()

            count = (count + 1) % lcm_cycles

            # Fix FPS.
            elapsed = (dt.datetime.now() - prev_time).total_seconds()
            sleep_duration = loop_cycle - elapsed
            if 0 < sleep_duration:
                time.sleep(sleep_duration)
            prev_time = dt.datetime.now()

    def _search_tweets(self, q, logged_ids: Set[tweet.ID], reporter_,
                       count=100, max_count_to_log=100) -> Set[tweet.ID]:
        """Searches for tweets and logs new tweets.

        Args:
            q: Query to search.
            logged_ids: DESTRUCTED. Already logged IDs. Filters new tweets.
            reporter_: Reporter for errors.
            count: The number of tweets to get. Less or equal to 100.
            max_count_to_log: The maximum number of new tweets to log. Less or
                equal to the count.

        Returns:
            New IDs logged.
        """
        nothing_to_do = count <= 0 or max_count_to_log <= 0
        if nothing_to_do:
            return set()

        # Request through the API.
        try:
            tweets = self._api.search(q=q, result_type="recent", count=count,
                                      tweet_mode="extended")
        except (error.TotalRateLimitError, tweepy.TweepError) as e:
            reporter_.report_error(e)
            return set()

        # Log only new tweets.
        def ids(tweets_): return set([t.id for t in tweets_])
        got_ids = ids(tweets)
        new_ids = got_ids - logged_ids
        if not new_ids:
            return set()
        new_tweets = [t for t in tweets if t.id in new_ids]
        new_tweets_to_log = new_tweets[:max_count_to_log]
        self._io.log_tweets(new_tweets_to_log)

        new_ids_logged = ids(new_tweets_to_log)
        logged_ids |= new_ids_logged
        return new_ids_logged

    def _track_values(self, ids_to_track: Deque[tweet.ID], reporter_,
                      num_ids_to_request=100) -> None:
        """Gets and logs various values for tweets.

        Args:
            ids_to_track: DESTRUCTED. All IDs to track.
            reporter_: Reporter for errors.
            num_ids_to_request: The number of IDs to request chosen from
                ids_to_track. Less or equal to 100.
        """
        if not ids_to_track:
            return

        # Determine IDs to track now.
        ids_to_track.rotate(num_ids_to_request)
        ids_to_track_now = list(
            itertools.islice(ids_to_track, 0, num_ids_to_request))

        # Request through the API.
        try:
            tweets = self._api.statuses_lookup(id_=ids_to_track_now,
                                               tweet_mode="extended")
        except (error.TotalRateLimitError, tweepy.TweepError) as e:
            reporter_.report_error(e)
            if Logger._is_retryable(e):
                ids_to_track.rotate(-num_ids_to_request)  # Revert.
        else:
            self._io.log_dynamics(tweets)

    def _track_retweets(self,
                        ids_to_track: Deque[tweet.ID],
                        ids_to_track_first: Set[tweet.ID],
                        logged_ids: Set[tweet.ID],
                        reporter_,
                        count=100) -> Optional[tweet.ID]:
        """Gets and logs retweets for a specified tweet.

        Args:
            ids_to_track: DESTRUCTED. All IDs to track.
            ids_to_track_first: DESTRUCTED. All IDs to track which have not been
                tracked yet.
            logged_ids: DESTRUCTED. Already logged IDs. Used to check all have
                been tracked and to log only new retweets.
            reporter_: Reporter for errors.
            count: The number of retweets to get. Less or equal to 100.

        Returns:
            ID given up continuing tracking. Note that this is not ID tracked.
        """
        if not ids_to_track:
            return

        # Determine an ID to request detail of retweets now.
        ids_to_track.rotate(1)
        id_to_track_now = ids_to_track[0]

        def give_up():
            ids_to_track.popleft()

        def revert():
            ids_to_track.rotate(-1)
            if is_first:
                ids_to_track_first.add(id_to_track_now)

        # For a tweet tracked first, check whether the number of retweets has
        # already exceeded the limit.
        is_first = id_to_track_now in ids_to_track_first
        if is_first:
            ids_to_track_first.remove(id_to_track_now)
            try:
                retweet_count = self._api.get_status(
                    id=id_to_track_now, tweet_mode="extended").retweet_count
            except (error.TotalRateLimitError, tweepy.TweepError) as e:
                reporter_.report_error(e, tweet_id=id_to_track_now)
                if Logger._is_retryable(e):
                    revert()
                    return
                else:
                    give_up()
                    return id_to_track_now
            else:
                if count < retweet_count:
                    give_up()
                    return id_to_track_now

        # Request through the API.
        try:
            tweets = self._api.retweets(id=id_to_track_now, count=count,
                                        tweet_mode="extended")
        except (error.TotalRateLimitError, tweepy.TweepError) as e:
            reporter_.report_error(e, tweet_id=id_to_track_now)
            if Logger._is_retryable(e):
                revert()
                return
            else:
                give_up()
                return id_to_track_now

        # Log the retweets if there are not overlooked retweets.
        got_ids = set([t.id for t in tweets])
        are_all_logged = bool(got_ids & logged_ids)
        if is_first or are_all_logged:
            # Log only new retweets.
            new_ids = got_ids - logged_ids
            if new_ids:
                new_tweets = [t for t in tweets if t.id in new_ids]
                self._io.log_retweets(new_tweets)
                logged_ids |= new_ids
        else:
            give_up()
            return id_to_track_now

    def inquire_viewers(self, tweet_id):
        """Inquires users who viewed a specified tweet to log."""
        tweets_df = self._io.read_tweets(index_col="id")
        tweeter = tweets_df.at[tweet_id, "user_id"]
        retweets_df = self._io.read_retweets(index_col="retweeted_id")
        retweeters = retweets_df.loc[tweet_id, "user_id"]
        ids_to_inquire = collections.deque(retweeters)
        ids_to_inquire.appendleft(tweeter)  # Inquire from the tweeter.

        reporter_ = reporter.Reporter(self._api)
        reporter_.report_beginning(f"Now inquiring viewers of {tweet_id}.")

        for id_to_inquire in ids_to_inquire:
            # Log followers' IDs of the tweeter/retweeter.
            cursor = -1
            while cursor != 0:
                try:
                    followers, (_, cursor) = self._api.followers_ids(
                        id=id_to_inquire, cursor=cursor)
                except error.TotalRateLimitError as e:
                    time.sleep(e.min_break_secs)
                except tweepy.TweepError as e:
                    reporter_.report_error(e, user_id=id_to_inquire)
                    if Logger._is_retryable(e):
                        continue
                    else:
                        break  # Give up this tweeter/retweeter.
                else:
                    self._io.log_followers(followers, id_to_inquire)

        reporter_.report_finish()

    @staticmethod
    def _is_retryable(e):
        if type(e) is error.TotalRateLimitError:
            return True
        if type(e) is tweepy.TweepError:
            tweep_error = e  # type: tweepy.TweepError
            if tweep_error.response is not None:
                response = tweep_error.response  # type: requests.Response
                if 500 <= response.status_code:  # Server is busy.
                    return True
                if 400 <= response.status_code:  # Something is wrong.
                    return False
        return False
