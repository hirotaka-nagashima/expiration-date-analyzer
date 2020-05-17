"""Manages data flow to evaluate classifiers."""

import os
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from sklearn import model_selection
from sklearn import naive_bayes

from analyzer import filters
from analyzer import timeextractor
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

    dataset = _load_dataset(dirs)
    all_y_pred = {}
    for additional_extension in ["bsig0.2ry0w101o0", "bsig0.9ry1w101o0"]:
        y_pred = _load_y_pred(dirs, additional_extension)
        all_y_pred[additional_extension] = y_pred
    _cross_validate_foreach(dataset, all_y_pred)


def _load_dataset(dirs) -> _Dataset:
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
        time_estimated = labeler.AutoLabeler.load_labels(dir_)
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


def _load_y_pred(dirs, additional_extension) -> List[sklearn.Y]:
    y_pred = []  # type: List[sklearn.Y]
    registered_ids = set()
    for dir_ in dirs:
        time_estimated = labeler.AutoLabeler.load_labels(
            dir_, additional_extension)
        for id_, y_pred_ in time_estimated.items():
            if id_ in registered_ids:
                continue
            y_pred.append(y_pred_)
            registered_ids.add(id_)
    return y_pred


def _cross_validate(dataset: _Dataset):
    x, y_pred, y_true = dataset
    text_divider = sklearn.TextDivider(top_k=243, pos_include={"その他", "フィラー", "感動詞", "記号", "形容詞", "助動詞", "接続詞", "接頭詞", "動詞", "副詞", "名詞", "連体詞"})
    sw = sklearn.Wrapper(naive_bayes.MultinomialNB(), text_divider)
    sw.cross_validate(x, y_pred, y_true=y_true)


def _cross_validate_foreach(dataset: _Dataset,
                            all_y_pred: Dict[str, List[sklearn.Y]]):
    x, _, y_true = dataset
    text_divider = sklearn.TextDivider(top_k=243, pos_include={"その他", "フィラー", "感動詞", "記号", "形容詞", "助動詞", "接続詞", "接頭詞", "動詞", "副詞", "名詞", "連体詞"})
    sw = sklearn.Wrapper(naive_bayes.BernoulliNB(), text_divider)
    for name, y_pred in all_y_pred.items():
        print(name, end="\t")
        sw.cross_validate(x, y_pred, y_true=y_true)


def _grid_search(dataset: _Dataset):
    x, y_pred, y_true = dataset
    x_train, x_test, y_pred_train, _, y_true_train, y_true_test = (
        model_selection.train_test_split(
            x, y_pred, y_true, random_state=0))

    text_divider = sklearn.TextDivider(top_k=243, pos_include={"その他", "フィラー", "感動詞", "記号", "形容詞", "助動詞", "接続詞", "接頭詞", "動詞", "副詞", "名詞", "連体詞"})

    # Finding parameters.
    best_mean = -np.inf
    best_param = None
    for param in [0.01, 0.05, 0.1, 0.5, 1.0]:
        print(param)
        sw = sklearn.Wrapper(naive_bayes.MultinomialNB(alpha=param),
                             text_divider=text_divider)
        mean = sw.cross_validate(x_train, y_pred_train, y_true=y_true_train)
        if best_mean < mean:
            best_param = param
            best_mean = mean

    # Final evaluation.
    print(best_param)
    sw = sklearn.Wrapper(naive_bayes.MultinomialNB(alpha=best_param),
                         text_divider=text_divider)
    sw.evaluate(x_train, x_test, y_pred_train, y_true_test)
