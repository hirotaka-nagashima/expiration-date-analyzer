from logger import fileio
from logger import logger
from twitter import authenticator


def main():
    def generate_logger(data_dir):
        auth = authenticator.Authenticator("./twitter_credentials.json")
        api = auth.authenticate()
        io = fileio.CSVHandler(data_dir)
        return logger.Logger(api, io)

    data_dir = "../data/production/ja"
    from estimator import commander
    commander.do(data_dir)


if __name__ == "__main__":
    main()
