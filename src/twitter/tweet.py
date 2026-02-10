ID = int


class Tweet:
    """Wrapper of an official tweet object."""

    def __init__(self, raw_tweet, shown_at):
        self.created_at = getattr(raw_tweet, "created_at")
        self.id: ID = getattr(raw_tweet, "id")

        # Attribute "text" or "full_text" is set according to the parameter
        # "tweet_mode" on a request.
        def replace_newlines(s, new):
            return new.join(s.splitlines())

        self.text: str | None = getattr(raw_tweet, "text", None)
        if self.text is not None:
            self.text = replace_newlines(self.text, " ")
        self.full_text: str | None = getattr(raw_tweet, "full_text", None)
        if self.full_text is not None:
            self.full_text = replace_newlines(self.full_text, " ")
        if self.text is None and self.full_text is None:
            message = "Both attributes 'text' and 'full_text' do not exist."
            raise AttributeError(message)

        entities = getattr(raw_tweet, "entities")

        def has(key):
            return key in entities and bool(entities[key])

        self.has_hashtags = has("hashtags")
        self.has_media = has("media")
        self.has_urls = has("urls")
        self.has_user_mentions = has("user_mentions")
        self.has_symbols = has("symbols")
        self.has_polls = has("polls")

        user = getattr(raw_tweet, "user")
        self.user_id: ID = user.id
        self.user_screen_name = user.screen_name
        self.user_followers_count = user.followers_count
        self.user_friends_count = user.friends_count
        self.user_verified = user.verified
        self.user_statuses_count = user.statuses_count

        self.retweet_count = getattr(raw_tweet, "retweet_count")
        self.favorite_count = getattr(raw_tweet, "favorite_count")

        retweeted = getattr(raw_tweet, "retweeted_status", None)
        self.retweeted_id: ID | None = getattr(retweeted, "id", None)

        self.shown_at = shown_at.replace(microsecond=0)
        self.elapsed_time = int((self.shown_at - self.created_at).total_seconds())
