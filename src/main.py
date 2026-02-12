from analyzer import timeextractor
from analyzer.labeler import labeler
from estimator import commander
from logger import fileio, logger
from twitter import authenticator


def main():
    credentials_path = "../credentials/twitter_credentials.json"
    data_dir = "../data/ja"
    tweets_lang = "ja"

    # 1. Logging of tweets
    auth = authenticator.Authenticator(credentials_path)
    api = auth.authenticate()
    io = fileio.CSVHandler(data_dir)
    log = logger.Logger(api, io)
    log.track_dynamics(
        loop_cycle=60,
        max_num_ids_to_track=100,
        logs_values=True,
        logs_retweets=False,
        lang=tweets_lang,
        min_retweets=50,
        additional_q="",
    )

    # 2. Estimation of expiration date using dynamics
    # 3. Estimation of expiration date using word co-occurrence
    timeextractor.extract_time_expressions(data_dir, lang=tweets_lang)  # preprocessing
    commander.do(data_dir)  # dataset construction (2), training & evaluation (3)

    # 4. To assist users to label tweets expiration date
    labeler.HandLabeler.run(data_dir)


if __name__ == "__main__":
    main()
