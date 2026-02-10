"""Manages data flow to evaluate classifiers."""

import os
from typing import List, Tuple

from sklearn import model_selection, naive_bayes

from analyzer import filters, timeextractor
from analyzer.labeler import labeler
from estimator import sklearn
from logger import fileio

_Dataset = Tuple[List[sklearn.X], List[sklearn.Y], List[sklearn.Y]]


def do(data_root_dir):
    files = sorted(os.listdir(data_root_dir))
    paths = [os.path.join(os.path.normpath(data_root_dir), f) for f in files]
    dirs = [p for p in paths if os.path.isdir(p)]

    # Prepare labels for training by estimation based on tweets' dynamics.
    for dir_ in dirs:
        labeler.AutoLabeler.run(dir_)

    dataset = _load_dataset(dirs, additional_extension="bsig0.9ry1w101o0")
    x, _, y_pred, _, y_true, _ = model_selection.train_test_split(
        *dataset, test_size=0.2, random_state=0
    )

    _cross_validate((x, y_pred, y_true))


def _load_dataset(dirs, additional_extension=None) -> _Dataset:
    """
    Args:
        dirs: Directories to load dataset.
        additional_extension: You can specify a version for a file created by
            AutoLabeler. For example, "time_estimated.json.bsig0" has an
            additional extension, "bsig0".
    """
    x = []  # type: List[sklearn.X]
    y_pred = []  # type: List[sklearn.Y]
    y_true = []  # type: List[sklearn.Y]
    registered_ids = set()
    for dir_ in dirs:
        # Load data from a disk.
        all_time_expressions = timeextractor.load_time_expressions(dir_)
        filter_ = filters.by_time_dependency(all_time_expressions)
        io = fileio.CSVHandler(dir_)
        tweets_df = io.read_tweets(index_col="id", filter_=filter_)
        time_estimated = labeler.AutoLabeler.load_labels(dir_, additional_extension)
        time_labeled = labeler.HandLabeler.load_labels(dir_)

        # Register data.
        for id_ in time_estimated:
            if id_ in registered_ids:
                continue
            text = tweets_df.at[id_, "full_text"]
            time_expressions = all_time_expressions[id_]
            x_ = (text, time_expressions)  # type: sklearn.X
            y_pred_ = time_estimated[id_]  # type: sklearn.Y
            y_true_ = time_labeled[id_]  # type: sklearn.Y

            x.append(x_)
            y_pred.append(y_pred_)
            y_true.append(y_true_)
            registered_ids.add(id_)

    return x, y_pred, y_true


def _cross_validate(dataset: _Dataset):
    x, y_pred, y_true = dataset
    text_divider = sklearn.TextDivider(
        top_k=729,
        pos_include={
            "その他",
            "フィラー",
            "感動詞",
            "記号",
            "形容詞",
            "助詞",
            "助動詞",
            "接続詞",
            "接頭詞",
            # "動詞",
            "副詞",
            "名詞",
            "連体詞",
        },
    )
    sw = sklearn.Wrapper(naive_bayes.BernoulliNB(), text_divider)
    sw.cross_validate(x, y_pred, y_true=y_true)
