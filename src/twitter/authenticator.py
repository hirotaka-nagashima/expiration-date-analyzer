import json
from typing import List

import tweepy

from twitter import api


class Credentials:
    """Data structure used only by the Authenticator."""

    def __init__(self, consumer_key, consumer_secret, access_token, access_secret):
        """
        Args:
            consumer_key: Consumer key for an app.
            consumer_secret: Consumer secret key for an app.
            access_token: Access token for a user.
            access_secret: Access secret token for a user.
        """
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_secret = access_secret


class Authenticator:
    """Authenticates us to get the api.API."""

    def __init__(self, credentials_path):
        self._credentials = []  # type: List[Credentials]
        self._load_credentials(credentials_path)

    def _load_credentials(self, path):
        """Loads Twitter API credentials."""
        with open(path) as file:
            raw_credentials = json.load(file)["credentials"]
        self._credentials.clear()
        for r in raw_credentials:  # each app
            self._credentials.append(
                Credentials(
                    consumer_key=r["consumerKey"],
                    consumer_secret=r["consumerSecret"],
                    access_token=r["accessToken"],
                    access_secret=r["accessSecret"],
                )
            )

    def authenticate(self) -> api.API:
        """Authenticates us and returns the api.API.

        Authenticates users and apps in the self._credentials and returns the
        api.API. We can also call this to re-authenticate on 89 errors: "Invalid
        or expired token".

        Returns:
            api.API defined in this package, not tweepy.API.
        """
        original_api = api.API()
        for c in self._credentials:  # each app
            user_auth = tweepy.OAuthHandler(c.consumer_key, c.consumer_secret)
            user_auth.set_access_token(c.access_token, c.access_secret)
            app_auth = tweepy.AppAuthHandler(c.consumer_key, c.consumer_secret)
            original_api.append(tweepy.API(user_auth), tweepy.API(app_auth))
        return original_api
